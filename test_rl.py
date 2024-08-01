import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import random 
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from utils import *
from REGAD.model import *
from REGAD.layers import *
from evaluation_AD import *
import arguments 
from base_detector.layer_gdn import GDN

args = arguments.parse_args()
torch.cuda.manual_seed(args.seed)

if args.run == 'cuda' and torch.cuda.is_available(): 
    device = torch.device('cuda:'+ str(args.cudano))
    print('cuda running...')
else:
    device = torch.device('cpu')
    print('cpu running...')
args.device = device
print(f'run on {args.device}')
print(f'run on {args.dataset}')


# Load data from dataset
adj, adj_norm, features, gnds = read_data(args.root_path, args.dataset)

train_indx_final, train_norm_indx, train_outlier_indx, valid_indx_final, test_indx_final, y_semi = \
    noisy_GAD_train_test_split_v1_40trn_20val_fixTrnAnom(
        gnds, args.seed, args.true_outliers_num, args.trn_ratio, args.val_ratio, args)
    
all_label = torch.from_numpy(y_semi).to(device) #noisy goround truth labels
true_label = torch.from_numpy(gnds) # real ground truth labels 
# matrix is sparse
adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
features = torch.FloatTensor(np.array(features)).to(device)
# train_batch_size = train_norm_indx.shape[0] *args.batch # too much is not efficient

# a hyperparameter needn't change
paces = np.linspace(0.1, 0.4, num=10)
noisy_label = copy.deepcopy(all_label)

# Model and optimizer initialization
base_model = GDN(nfeat=features.shape[1],
                nhid=args.hidden,
                dropout=args.dropout)
gdn_path = args.model_path + args.dataset + '_gdn.pkl'
base_model.load_state_dict(torch.load(gdn_path, map_location=device))
base_model = base_model.to(device)

# define the basic components
best_pace, noisy_candi, anomaly_tensor, norm_tensor, nodes_embeddings, anomaly_score = process_test_data(
    args.dataset, base_model, features, adj_tensor, all_label, paces, args.rate_ano_sel)
edge_mask = get_edge_mask(adj_tensor, noisy_candi)
noisy_candi = torch.tensor(noisy_candi)
all_label, anomaly_changes, norm_changes = purify_labels(all_label, anomaly_tensor, norm_tensor)
print('The best pace for {:02f}'.format(best_pace), 'of dataset {}'.format(args.dataset))
print( 'The max possible edges count {}'.format(torch.count_nonzero(edge_mask))) 
print('The anomaly changed nodes are {}'.format(anomaly_changes), 'and the normal changes are {}'.format(norm_changes))
print('The length of noisy_candi is {}'.format(noisy_candi.shape))

policy = PolicyNetwork(nodes_embeddings, args.hidden_policy).to(device) #.half()
policy_optimizer = torch.optim.Adam(
    policy.parameters(), lr=args.lr_policy, weight_decay=args.weight_policy)

env = Environment(nodes_embeddings, adj_tensor, base_model)


args.cut_path = args.dataset + '_num_episodes_' + str(args.num_episodes) + '_samples_' + str(args.num_samples) \
                +  '_lr_policy_' + str(args.lr_policy) + '_ano_rate_'+ str(args.rate_ano_sel) \
                    +'_epoch_'+str(args.num_epochs) +'_true_outlier_'+ str(args.true_outliers_num)+'_of_'+str(args.max_labeled_outliers_num)+'_seed_' + str(args.seed) 

save_model_path = args.model_path + args.dataset + '_cutrl.pkl'

def train_rl(env, model, optimizer, features, adj_tensor, edge_mask, noisy_candi, all_label, args = args):
    train_loss_list = []
    max_valid_auc = 0
    min_trn_loss = 100
    cut_list_epoch = []
    
    start_time = time.time()
    update_flag = False 
    
    for epoch in range(args.num_epochs):
        model.train()         
        
        loss_episode, cut_num_list, state, pred_scores = \
            train_policy_net(env, model, optimizer, edge_mask, noisy_candi, features, all_label, args)
        
        current_loss = np.mean(loss_episode)
        train_loss_list.append(current_loss)
        current_cut = np.mean(cut_num_list)
        cut_list_epoch.append(current_cut)
        
        if epoch == args.epochs-1:
            print('Epoch: {:04d}'.format(epoch), 'Training_Loss: {:.4f}'.format(
                current_loss), 'time: {:.4f}s'.format(time.time() - start_time), \
                    'Cutnum: {:.4f}'.format(current_cut) )
        
        # save hyparameters
        if current_loss < min_trn_loss:
            min_trn_loss = current_loss
            # torch.save(model.state_dict(), save_model_path)
            update_flag = False 
        else: 
            update_flag = True
            
        # a new version of candidates and confident sets
        if update_flag:
            best_pace, noisy_candi, anomaly_list, norm_list, nodes_embeddings, anomaly_score = process_test_data(
                    args.dataset, base_model, features, adj_tensor, all_label, paces, args.rate_ano_sel, 
                        path = gdn_path)
            edge_mask = get_edge_mask(noisy_candi, adj_tensor)
            all_label, anomaly_changes, norm_changes = purify_labels(all_label, anomaly_list, norm_list)
        
        with torch.no_grad():
            # 假设你有一个函数来处理验证数据并计算损失
            # total_reward, all_rewards, all_actions, state =  test_policy_net(env, policy, edge_mask, noisy_candi, features, all_label, args)
            # anomaly_score, embed_be = base_model(features, state[1])
            pred_anomaly_score = pred_scores[valid_indx_final]
            y_true = true_label.numpy()[valid_indx_final]
            results_metric_AD = szhou_AD_metric(pred_anomaly_score, y_true)
            if results_metric_AD[-2] > max_valid_auc: 
                torch.save(model.state_dict(), save_model_path)
                max_valid_auc = results_metric_AD[-2]
        
    end_time = time.time()
    total = end_time - start_time
    print('The total time of training several epochs are {:.4f}'.format(total))
    
    return cut_list_epoch, state


cut_list_epoch, state = train_rl(env, policy, policy_optimizer, features, adj_tensor, edge_mask, noisy_candi, all_label) 
print('The training process has been finished !')


#test 
total_reward, all_rewards, all_actions, state =  test_policy_net(env, policy, edge_mask, noisy_candi, features, all_label, args)
anomaly_score, embed_be = base_model(features, state[1])
pred_anomaly_score = anomaly_score.detach().cpu().numpy()[test_indx_final]
y_true = true_label.numpy()[test_indx_final]
results_metric_AD = szhou_AD_metric(pred_anomaly_score, y_true)
df = pd.DataFrame(np.array(results_metric_AD).reshape(-1, 14),
                    columns=['Precision_100', 'Precision_200', 'Precision_300', 'Precision_400', 'Precision_500', 'Precision_600',
                            'Recall_100', 'Recall_200', 'Recall_300', 'Recall_400', 'Recall_500', 'Recall_600', 'AUC', 'AUPR'])
df.to_csv(args.test_path + 'cut_results_' + args.cut_path + '.csv')

print("RL process finished!!")