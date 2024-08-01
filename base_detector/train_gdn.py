import time
import os
print(os.getcwd())
import random 
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
# important settings for modules
import sys
sys.path.append('./')
current_path = os.getcwd()
os.environ['DEFAULT_PATH'] = current_path
from base_detector.layer_gdn import *
from evaluation_AD import *
import arguments 
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
args = arguments.parse_args()
if args.run == 'cuda' and torch.cuda.is_available():
    args.cuda = True
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda', index=args.cudano)
    print('cuda running...')
else:
    args.cuda = False
    device = torch.device('cpu')
    print('cpu running...')
args.device = device
# print(f'run on {args.device}')
# print(f'run on {args.dataset}')
# args.version =2
# print(f'run on {args.version}')
# if args.root_path == './data/noisy_dataset_v1/':
#     args.version =1 
# else:
#     args.version =2


# Load data from dataset
adj, adj_norm, features, gnds = read_data(args.root_path, args.dataset)

train_indx_final, train_norm_indx, train_outlier_indx, valid_indx_final, test_indx_final, y_semi = \
    noisy_GAD_train_test_split_v1_40trn_20val_fixTrnAnom(
        gnds, args.seed, args.true_outliers_num, args.trn_ratio, args.val_ratio, args)

# from numpy to tensor (different ground truth labels for training and testing)
# use y_semi, as unknown_outlier are set gnd = 0
all_label = torch.from_numpy(y_semi).to(device) #noisy goround truth labels
true_label = torch.from_numpy(gnds).to(device) # real ground truth labels 

# the parameters after some processing 
adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
# adj_tensor_den = adj_tensor.to_dense()
features = torch.FloatTensor(np.array(features)).to(device)
train_batch_size = train_norm_indx.shape[0] *2 # too much is not efficient


# Model and optimizer initialization
GDN_model = GDN(nfeat=features.shape[1], nhid=args.hidden,
                dropout=args.dropout)
GDN_model = GDN_model.to(device)
optimizer = optim.Adam(GDN_model.parameters(),
                       lr=args.lr_gdn, weight_decay=args.weight_decay)
# to save the model
learned_model_path = args.model_path + args.dataset + '_gdn.pkl'
args.gdn_path = learned_model_path
# 
def train_epoch(epoch, GDN_model, optimizer, batch_index):
    GDN_model.train()
    optimizer.zero_grad()
    pred_score, nodes_embeddings = GDN_model(features, adj_tensor)
    # compute categories based on score
    # only use train data to get train loss
    trn_loss = deviation_loss_torch(
        all_label[batch_index], pred_score[batch_index])  #since 这部分模拟不确定noisy label获得high confident list
    trn_loss.backward()
    optimizer.step()
    return trn_loss.item()

def train_GDN(GDN_model):
    
    min_trn_loss = 100    
    trn_loss_list = []
    max_valid_auc = 0.0
    valid_auc_list = []
    epoch_j = 0
    patience_counter =0     
    
    for trn_indices in train_batch_iter_v2_FullDataTrn(train_outlier_indx, train_norm_indx, train_batch_size, args.epochs, seed=args.seed):
        t = time.time()
        trn_loss = train_epoch(epoch_j, GDN_model, optimizer, trn_indices)
        trn_loss_list.append(trn_loss)
        
        print('Epoch: {:04d}'.format(epoch_j + 1), 'Training_Loss: {:.4f}'.format(
                trn_loss), 'time: {:.4f}s'.format(time.time() - t))
        current_loss = trn_loss

        # save hyparameters
        if current_loss < min_trn_loss:
            min_trn_loss = current_loss
            torch.save(GDN_model.state_dict(), learned_model_path)
            # print('save current model...')
            print(f"Epoch {epoch_j}: Train loss improved, saving model...")
        else:
            patience_counter += 1
            print(f"Epoch {epoch_j}: No improvement in training loss for {patience_counter} epochs...")
        
        if patience_counter >= args.patient:
            print("Early stopping triggered. Stopping training...")
            break
        epoch_j += 1
    
    # validation process 
    GDN_model.eval() 
    final_pred_score, nodes_embeddings = GDN_model(features, adj_tensor)
    anomaly_score = final_pred_score.detach().cpu().numpy()
    test_y_score = anomaly_score[valid_indx_final]  
    y_noisy_true = all_label.detach().cpu().numpy()
    test_truth = y_noisy_true[valid_indx_final]
    results_metric_AD = szhou_AD_metric(test_y_score, test_truth)
    df = pd.DataFrame(np.array(results_metric_AD).reshape(-1, 14),
                      columns=['Precision_100', 'Precision_200', 'Precision_300', 'Precision_400', 'Precision_500',
                               'Precision_600',
                               'Recall_100', 'Recall_200', 'Recall_300', 'Recall_400', 'Recall_500', 'Recall_600', 'AUC',
                               'AUPR'])
    df.to_csv(args.val_path + 'vali_gdn_' + args.save_path + '.csv')
    if results_metric_AD[-2] > max_valid_auc: 
        torch.save(GDN_model.state_dict(), learned_model_path) 
        print('save the best model')
        valid_auc_list.append(results_metric_AD[-2])
        max_valid_auc = results_metric_AD[-2]
    return nodes_embeddings
        
    
def test_GDN(GDN_model, test_indx_final):
    GDN_model.load_state_dict(torch.load(learned_model_path))
    GDN_model.eval()
    final_pred_score, nodes_embeddings = GDN_model(features, adj_tensor)
    anomaly_score = final_pred_score.detach().cpu().numpy()
    
    y_true = all_label.detach().cpu().numpy()  # nosiy ones still
    test_y_score = anomaly_score[test_indx_final]  
    test_truth = y_true[test_indx_final]
    
    results_metric_AD = szhou_AD_metric(test_y_score, test_truth)
    df = pd.DataFrame(np.array(results_metric_AD).reshape(-1, 14),
                      columns=['Precision_100', 'Precision_200', 'Precision_300', 'Precision_400', 'Precision_500',
                               'Precision_600',
                               'Recall_100', 'Recall_200', 'Recall_300', 'Recall_400', 'Recall_500', 'Recall_600', 'AUC',
                               'AUPR'])
    df.to_csv(args.test_path + 'test_gdn_' + args.save_path + '.csv')
    
    return final_pred_score, y_true, nodes_embeddings
    
# finish train and validation
args.run = 5
args.save_path = args.dataset + '_outliers_' + str(args.true_outliers_num)+ '_seed_' + str(args.seed)
start = time.time()
for i  in  range(args.run): 
    node_emb = train_GDN(GDN_model)
end = time.time()
print('total time: {:.4f}s'.format(end - start))
print('Finish validation!!! ')

# paces = np.linspace(0.1, 0.5, num=10)
paces = np.linspace(0.1, 0.5, num=10)
anomaly_score, y_true, node_emb = test_GDN(GDN_model, test_indx_final)
# noisy_can = select_noisy_candidates(args.num_rate, anomaly_score).to(device)
best_pace, candi_auc, noisy_candi = epsilon_greedy(args.dataset, anomaly_score, all_label, paces, epsilon=0.02, iterations=100)
anomaly_list, norm_list = select_confident(anomaly_score, args.rate_ano_sel)
print("Preparation for RL is finished!!")