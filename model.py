import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import gc
from torch.cuda.amp import autocast, GradScaler
from REGAD.layers import *

# policy network definition
class PolicyNetwork(nn.Module):
    def __init__(self, embedding, hidden_size): # hidden is the hyper
        super(PolicyNetwork, self).__init__()
        self.embed_size = embedding.shape[1] # 64 
        self.num_nodes = embedding.shape[0] # node number
        self.gc1 = GraphConvolution(self.embed_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, self.num_nodes)
        

    def forward(self, x, adj, edge_mask, dropout):
        x = F.leaky_relu(self.gc1(x, adj), negative_slope=0.01)
        x = F.dropout(x, dropout)
        x = self.gc2(x, adj)
        action_probs = F.softmax(x, dim=1)
        self.masked_action_probs = action_probs * edge_mask # need it as dense matrix
        return self.masked_action_probs
    
    # transite from probilities to actions 
    # actions are represented by cut_mask matrix
    def get_action(self, anomaly_candidate, num_samples): #anomaly list refers to candidates
        # 对角线 = 0, keep the self loop, not include this edge into actions 上三角矩阵保留
        probs_adj = self.masked_action_probs.clone()
        probs_adj.fill_diagonal_(0)
        upper_triangular = torch.triu(probs_adj)
        
        # Get the indices of the top-k probabilities
        _, topk_indices = torch.topk(upper_triangular, num_samples, dim=1)
        # Convert the top-k indices to a list of (head, tail) tuples
        action_tuple_list = []
        for i in anomaly_candidate:
            for j in topk_indices[i]:
                action_tuple_list.append((i.item(), j.item()))
                        
        return action_tuple_list 

# environment which provides the action spaces 
class Environment:
    def __init__(self, node_embeddings, adj_matrix, base_detector):
             
        self.node_embeddings = node_embeddings
        self.adj_matrix = adj_matrix.to_dense()
        self.base_detector = base_detector  
         
        # the initial state should not be changed      
        self.ini_state = (self.node_embeddings, self.adj_matrix)

    # def to(self, device):
    #     self.node_embeddings = self.node_embeddings.to(device)
    #     self.adj_matrix = self.adj_matrix.to(device)
    #     self.base_detector = self.base_detector.to(device)
    #     self.device = device
    #     # 确保所有其他模型或张量也转移到了device
    #     return self
    
    # if the action is taken what is the reward?
    def step(self, state, action, all_label, node_features, edge_mask): #action is list of edges(start,end)
        num = 0
        current_emb, adj_before = state
        edge_mask_copy = edge_mask.clone() 
        modified_adj = adj_before.clone()
        cut_mask = torch.zeros(node_features.shape[0],node_features.shape[0])
        # unique_edges = set()
        # for edge in action:
        #     unique_edges.add((edge[0], edge[1]))
        # print("Unique edges in action:", len(unique_edges))
        invaild_num = 0
        for edge in action:
            if modified_adj[edge[0], edge[1]] != 0: # 检查边是否存在,存在才会进行cut
                modified_adj[edge[0], edge[1]] = 0
                modified_adj[edge[1], edge[0]] = 0
                num+=1
                cut_mask[edge[0], edge[1]] = 1 
                cut_mask[edge[1], edge[0]] = 1
                edge_mask_copy[edge[0], edge[1]] = 0
                edge_mask_copy[edge[1], edge[0]] = 0
                # print(f"Modifying edge: {edge}")
            else:
                invaild_num +=1
                # print(f"Edge already zero or does not exist: {edge}")
        # assert len(unique_edges) == invaild_num + num, "The total edges are correct."
        # print('The cut edges num equals to the cutmask matrix')
        # print(torch.where(cut_mask==1)[0].shape[0] == 2*num) 
        
        # all_label and new_embeddings are tensors
        with torch.no_grad():
            all_label_np = all_label.cpu().detach().numpy()
            self.base_detector.eval()
            pred_be, embed_be = self.base_detector(node_features, adj_before)
            pred_be = pred_be.cpu().detach().numpy()
            pred_af, embed_af = self.base_detector(node_features, modified_adj)
            pred_af = pred_af.cpu().detach().numpy()
        
            auc_after = roc_auc_score(all_label_np, pred_af)
            auc_before = roc_auc_score(all_label_np, pred_be)
        
        reward = (auc_after - auc_before) * 1e3
        if reward > 0.5 or reward < -0.5:
            reward = 0 
        
        reward = torch.tensor(reward)
        # torch.cuda.empty_cache() 
        return (embed_af, modified_adj), reward, num, edge_mask_copy, edge_mask, pred_af 

    def reset(self):
        embeddings, adj = self.ini_state
        return embeddings, adj

   
def train_policy_net(env, policy, optimizer, edge_mask, anomaly_list, node_features, all_label, args):        
    num_episodes = args.num_episodes
    num_samples = args.num_samples
    
    cut_num_list = []
    # all_log_probs = []
    # all_rewards = [] # to record rewards in all episodes
    policy_loss_list = []
    if args.dataset == 'amazon_electronics_photo':
        thres = 10
    else: 
        thres = 20
    
    scaler = GradScaler()  # 初始化梯度缩放器
    # every episode is a complete makov process
    for episode in range(num_episodes):
        optimizer.zero_grad()
        state = env.reset()
        done = False
        episode_rewards = [] 
        episode_log_probs = []
        # logs for this episode's details
        device_cpu = torch.device("cpu")
        total_num = 0
        
        while not done:    
            with autocast():
                action_probs = policy(state[0], state[1], edge_mask, args.dropout_policy).clamp(min=1e-8)
                action = policy.get_action(anomaly_list, num_samples)
                next_state, reward, num, cut_mask, edge_mask, pred_scores = env.step(state, action, all_label, node_features, edge_mask)
            print('cut edges:{}'.format(num))
            total_num += num            
            
            log_prob = torch.log(torch.sum(torch.mul(action_probs, cut_mask.to(args.device))))  # record all probs of all edges
            # 记录奖励和动作的对数概率
            if num < thres:
                done = True   
                continue             
            else:
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                state = next_state        
            
            if total_num >= 0.5*len(anomaly_list): 
                done = True   
                continue     
        
        cut_num_list.append(total_num)
        # all_log_probs.append(torch.stack(episode_log_probs).sum())
        # all_rewards.append(torch.stack(episode_rewards).sum())

        # 策略梯度更新
        policy_loss = [] # not very easy if reinforce 
        G = 0
        gamma = 0.99 
        returns = []
        
        # 从后向前计算累积回报
        for reward in reversed(episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)    
        returns = torch.tensor(returns).to(device_cpu)
        assert len(episode_log_probs) == len(returns), \
                "The lengths of log_probs and returns must match."
        for log_prob, Gt in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob.to(device_cpu) * Gt)  # 直接使用张量操作，不转换为item 
        
        policy_loss = torch.stack(policy_loss).sum().to(args.device)     
        print('Episode {} is finished'.format(episode), 'The cut edges are {}'.format(total_num))
        
        if args.run == 'cpu':
            policy_loss.backward(retain_graph=True) #
            optimizer.step()
        else: 
            scaler.scale(policy_loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        policy_loss_list.append(policy_loss.item())
        
        torch.cuda.empty_cache()  # 清空未使用的缓存
        gc.collect()
        
    return policy_loss_list, cut_num_list, state, pred_scores #, all_rewards, all_log_probs
    

def test_policy_net(env, policy, edge_mask, anomaly_list, node_features, all_label, args):
    num_samples = args.num_samples
    if args.dataset == 'amazon_electronics_photo':
        thres = 10
    else: 
        thres = 20
    state = env.reset()
    done = False
    total_num = 0
    all_rewards = []
    all_actions = []
    
    while not done:
        action_probs = policy(state[0], state[1], edge_mask, args.dropout_policy).clamp(min=1e-8)
        action = policy.get_action(anomaly_list, num_samples)
        next_state, reward, num, cut_mask, edge_mask, pred_scores = env.step(state, action, all_label, node_features, edge_mask)
        total_num += num
               
        state = next_state
        
        if num <thres or total_num >= 0.5*len(anomaly_list):
            done = True
        else: 
            all_rewards.append(reward)
            all_actions.append(action)
    
    # 计算总奖励
    total_reward = sum(all_rewards)
    
    return total_reward, all_rewards, all_actions, state
    

"""
the completed GDN model
"""
# the definition of cut edges 
class GDN_cut(nn.Module):
    def __init__(self, policy_net, embed_model, valuator, policy_optimizer):
        super(GDN_cut, self).__init__()
        self.policy = policy_net
        self.embed_model = embed_model
        self.Outlier_Valuator = valuator
        self.optimizer = policy_optimizer
        

    def forward(self, nodes, adj, edge_mask, anomaly_list, all_label, args):        
        embeddings = self.embed_model(nodes, adj, args)
        # action_probs = self.policy(embeddings, adj, edge_mask)
        # action_tuple_list = self.get_action(action_probs)
        self.env = Environment(nodes, embeddings, adj, self.embed_model, self.Outlier_Valuator)
        num_episodes = args.num_episodes
        num_samples = args.num_samples
        (final_embeddings, final_adj), cut_num_list,cut_num = self.train_policy(embeddings, adj, edge_mask, anomaly_list, num_samples, num_episodes, all_label, args)
        # Calculate loss using final_embeddings and labels
        pred_score = self.Outlier_Valuator(final_embeddings)

        return pred_score,final_embeddings, final_adj, cut_num_list,cut_num
    
    def test_gdn_cut(self, nodes, adj, edge_mask, anomaly_list, all_label, args):
        embeddings = self.embed_model(nodes, adj, args)
        # action_probs = self.policy(embeddings, adj, edge_mask)
        # action_tuple_list = self.get_action(action_probs)
        self.env = Environment(nodes, embeddings, adj, self.embed_model, self.Outlier_Valuator)
        num_samples = args.num_samples
        final_embeddings, final_adj = self.inference_policy(embeddings, adj, edge_mask, anomaly_list, num_samples, all_label, args)
        # Calculate loss using final_embeddings and labels
        pred_score = self.Outlier_Valuator(final_embeddings)
        return pred_score
    
    def inference_policy(self, embeddings, adj, edge_mask, anomaly_list, num_samples, all_label, args):
        state = self.env.reset()
        done = False
        cut_num = 0
        while not done:
            action_probs = self.policy(state[0], state[1], edge_mask, args)
            num_nodes = adj.shape[0]
            action, cut_mask = self.get_action(action_probs, anomaly_list, num_nodes, num_samples)
            cut_mask = cut_mask.to(args.device)

            next_state, reward, num = self.env.step(action, cut_mask, num_samples, all_label, args)
            cut_num += num
            state = next_state
            if cut_num >= 0.5*len(anomaly_list) or num == 0: 
                done = True
        return state  # return final embeddings and adjacency
    
    #the definition of function in class include self***
    def get_action(self, action_probs, anomaly_list, num_nodes, num_samples):
        # Flatten the non-zero values of the matrix to a vector
        # 先变成上三角阵，然后元素非零的action probs
        action_probs.fill_diagonal_(0)
        upper_triangular = torch.triu(action_probs)
        #action_probs_vector = upper_triangular[action_probs != 0]
        # Initialize cut_mask as a matrix of ones
        cut_mask = torch.ones((num_nodes, num_nodes))
        # Get the indices of the top-k probabilities
        _, topk_indices = torch.topk(upper_triangular, num_samples, dim=1)

        # Convert the top-k indices to a list of (head, tail) tuples
        action_tuple_list = []
        for i in anomaly_list:
            for j in topk_indices[i]:
                action_tuple_list.append((i.item(), j.item()))
                # Set the corresponding positions in cut_mask to 0
                cut_mask[i, j] = 0
                cut_mask[j, i] = 0
        
        return action_tuple_list, cut_mask
    
    def train_policy(self, embeddings, adj, edge_mask, anomaly_list, num_samples, num_episodes, all_label, args):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            cut_num_list = []
            cut_num = 0
            while not done:
                action_probs =self.policy(state[0], state[1], edge_mask, args)
                num_nodes = adj.shape[0]
                action, cut_mask = self.get_action(action_probs, anomaly_list, num_nodes, num_samples)
                cut_mask = cut_mask.to(args.device)
                
                next_state, reward, num = self.env.step(action, cut_mask, num_samples, all_label, args)
                cut_num_list.append(num)
                cut_num += num
                cut_mask_re = torch.logical_not(cut_mask).int().to(args.device)
                loss = -torch.log(torch.sum(torch.mul(action_probs,cut_mask_re))) * reward
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                state = next_state
                if cut_num >= 0.5*len(anomaly_list) or num == 0: 
                    done = True
        return state,cut_num_list,cut_num  # return final embeddings and adjacency matrix
    

class GDN_com(GDN_cut):
    def __init__(self, policy_net, embed_model, valuator, valuator_rl, policy_optimizer):
        super().__init__(policy_net, embed_model, valuator, policy_optimizer)
        self.Valuator_aug = valuator_rl
    
    def forward(self, nodes, adj, edge_mask, noisy_list, anomaly_list, norm_list, all_label, args, train_flag):        
        embeddings = self.embed_model(nodes, adj, args)
        # action_probs = self.policy(embeddings, adj, edge_mask)
        # action_tuple_list = self.get_action(action_probs)
        self.env = Environment(nodes, embeddings, adj, self.embed_model, self.Outlier_Valuator)
        num_episodes = args.num_episodes
        num_samples = args.num_samples
        (final_embeddings, final_adj), cut_num_list,cut_num = self.train_policy(embeddings, adj, edge_mask, noisy_list, num_samples, num_episodes, all_label, args)
        # Calculate loss using final_embeddings and labels
        # pred_score = self.Valuator_aug(final_embeddings)
        mask_index, thresholds, pred_score = self.Valuator_aug(final_embeddings, nodes, all_label, anomaly_list, norm_list, args, train_flag)


        return mask_index, thresholds, pred_score
    
    def loss(self, nodes, adj, edge_mask, noisy_list, anomaly_list, norm_list, all_label, args, train_flag = True):
        mask_index, thresholds, pred_score = self.forward(nodes, adj, edge_mask, noisy_list, anomaly_list, norm_list, all_label, args, train_flag)
        
        # get predctions of scores and labels
        # score_proba = nn.functional.softmax(pred_score, dim=1)
        # _, label_proba = torch.max(score_proba, dim=1, )

        # transform list into set
        # use index set as list again to change labels
        labels_re = all_label.clone()
        set_index = list(set(
            element for sublist in mask_index for element in sublist))
        for i in range(0, len(set_index)):
            if labels_re[set_index[i]] == 0:
                labels_re[set_index[i]] = 1

        return labels_re, mask_index, pred_score
    
    def test_gdn_cut(self, nodes, adj, edge_mask, noisy_list, anomaly_list, norm_list, all_label, args, train_flag = False):
        embeddings = self.embed_model(nodes, adj, args)
        # action_probs = self.policy(embeddings, adj, edge_mask)
        # action_tuple_list = self.get_action(action_probs)
        self.env = Environment(nodes, embeddings, adj, self.embed_model, self.Outlier_Valuator)
        num_samples = args.num_samples
        final_embeddings, final_adj = self.inference_policy(embeddings, adj, edge_mask, noisy_list, num_samples, all_label, args)
        # Calculate loss using final_embeddings and labels
        mask_index, thresholds, pred_score = self.Valuator_aug(final_embeddings, nodes, all_label, anomaly_list, norm_list, args, train_flag)

        labels_re = all_label.clone()
        set_index = list(set(
            element for sublist in mask_index for element in sublist))
        for i in range(0, len(set_index)):
            if labels_re[set_index[i]] == 0:
                labels_re[set_index[i]] = 1
        return labels_re, mask_index, pred_score



#augment part is not run in the policy in the loop
class GDN_aug(nn.Module):
    def __init__(self, nfeat, nhid, ano_count, args, dropout):
        super(GDN_aug, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)  # 2*64
        self.gc2 = GraphConvolution(2 * nhid, nhid)  # 64
        self.dropout = dropout
        self.embeddings = None
        self.Outlier_Valuator = RL_Valu(nfeat, nhid, ano_count, args)
        # self.Outlier_Valuator = nn.Sequential(
        #     nn.Linear(nhid, 512),  # same as Kaize paper
        #     nn.ReLU(True),
        #     nn.Linear(512, 1)
        # )
        # self.xent = nn.CrossEntropyLoss()

    def forward(self, nodes, labels, adj, anomaly_list, norm_list, args, train_flag=True):
        nodes = F.relu(self.gc1(nodes, adj))
        nodes = F.dropout(nodes, self.dropout, train_flag)
        embeddings = self.gc2(nodes, adj)

        mask_index, thresholds, pred_score = self.Outlier_Valuator(embeddings, nodes, labels,
                                                                   anomaly_list, norm_list, args)

        return mask_index, thresholds, pred_score

    def loss(self, nodes, labels, adj, anomaly_list, norm_list, args, train_flag=True):
        mask_index, thresholds, pred_score = self.forward(
            nodes, labels, adj, anomaly_list, norm_list, args, train_flag)
        
        # get predctions of scores and labels
        # score_proba = nn.functional.softmax(pred_score, dim=1)
        # _, label_proba = torch.max(score_proba, dim=1, )

        # transform list into set
        # use index set as list again to change labels
        labels_re = labels.clone()
        set_index = list(set(
            element for sublist in mask_index for element in sublist))
        for i in range(0, len(set_index)):
            if labels_re[set_index[i]] == 0:
                labels_re[set_index[i]] = 1

        return labels_re, mask_index, pred_score
        '''
        # the initial loss computation
        label_loss = self.xent(
            pred_score.squeeze()[batch_index], labels_re[batch_index])

        return labels_re, label_loss
        '''