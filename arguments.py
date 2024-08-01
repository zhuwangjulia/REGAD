import argparse

def parse_args():

    """
        training model with data augmentation 
    """

    parser = argparse.ArgumentParser()

    # hyper for gpu
    parser.add_argument('--run', type=str, default='cuda',
                        help='cpu or cuda CUDA training')
    parser.add_argument('--cudano', type=int, default=1 , help='CUDA TO USE.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--test', type=str, default="test",
                        help='For validation or test')
    # hyper for paths
    parser.add_argument("--val_path", type=str, default="./results/")
    parser.add_argument("--val_plot", type=str, default="./results_plot/")
    parser.add_argument("--test_path", type=str, default="./results_test/")
    parser.add_argument("--test_plot", type=str, default="./results_test_plot/")
    parser.add_argument("--model_path", type=str, default="./models_save/")
    parser.add_argument("--root_path", type=str,
                        default="./data/noisy_dataset_v2/")
    
    # hyper for datasets
    parser.add_argument('--dataset', type=str, default='amazon_electronics_photo',
                        choices=['ACM_Small_SideInfo_szhou', 'Amazon_clothing',
                                'amazon_electronics_computers', 'amazon_electronics_photo',
                                'ms_academic_cs'])
    parser.add_argument('--true_outliers_num',
                        help='outliers with true labels', type=int, default=15)
    parser.add_argument('--max_labeled_outliers_num', type=int,
                        help='all the known anomalies on hands', default=30)
    parser.add_argument('--trn_ratio', type=float,
                        default=0.4, help='train data ratio')
    parser.add_argument('--val_ratio', type=float,
                        default=0.2, help='valid data ratio')
    parser.add_argument('--tes_ratio', type=float,
                        default=0.4, help='test data ratio')
    # params for GDN
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epoch to train.')
    parser.add_argument('--patient', type=int, default=50,
                        help='Number of epoch for patience on early_stop.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units (for all 234).')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of GNN layers, 2/3/4.')
    parser.add_argument('--lr_gdn', type=float, default=0.0005,
                        help='Initial learning rate for all dataset is 0.0005.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')  # no dropout     
    parser.add_argument('--batch', type=float, default=2,
                        help='to get balanced batchsize.')  # no dropout
    
    # parameters for rl part
    parser.add_argument('--hidden_k', type=int,
                        default=15, help='Number of neighbors for anomalies in the hidden space'
                        + '10 15 20 25')
    parser.add_argument('--r_r', type=float,
                        default=0.5, help='Number of neighbors for anomalies in the hidden space'
                        + '0.5,0.6,0.7,0.75')    
    parser.add_argument('--step-size', type=float,
                        default=0.005, help='RL action step size'
                        + ' 0.01 0.005 0.001')
    parser.add_argument('--ini_th', type=float,
                        default=0.90, help='the first threshold'
                        + '0.95 0.90 0.85')
    parser.add_argument('--episodes', type=int,
                        default=50, help='Number of neighbors for anomalies in the hidden space'
                        + '10 20 30 50')
    
    # hyper for cut edges part
    # hyper-parameters for rl 
    parser.add_argument('--num_epochs', type=int,
                        default=1, help='Number of epochs.')  
    parser.add_argument('--rate_ano_sel', type=float,
                        default=0.005, help='Proportion of anomaly selected as high-confidence in the rank list'
                        + '0.001 max')
    parser.add_argument('--num_samples', type=int, default=120,
                        help=' cut edges once for each anomaly node')
    # parser.add_argument("--num_rate", type=float, default=0.1,
    #                     help=" 0.1 this is for std(the largetest number is all edges in the subgraph)")
    parser.add_argument('--num_episodes', type=int, default=5,
                        help=' the policy network epi')
    parser.add_argument('--hidden_policy', type=int, default=128,
                        help='Number of hidden units for policy network.')
    parser.add_argument('--lr_policy', type=float, default=0.005,
                        help='learning rate for policy network.')
    parser.add_argument('--weight_policy', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout_policy', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')  # no dropout
    return parser.parse_args()

