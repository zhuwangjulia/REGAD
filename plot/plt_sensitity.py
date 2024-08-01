import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def plot_hyperparameters_analysis(datasets, hyperparameter_results,x,hyper_names):
    
    # 创建一个图和子图
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(5, 3.5))  # 注意这里的变化
    
    
    #  对每个数据集画一条线
    ax2.plot(x, hyperparameter_results[0], color='blue',marker='o', label=datasets[0])
    max_y = max(hyperparameter_results[0])
    max_x = x[hyperparameter_results[0].index(max_y)]    
    ax2.scatter([max_x], [max_y], color='red', marker='*', s=100)  # 标记最高点
    ax2.set_ylim(0.4,0.7)
    # ax2.legend()
    
    ax1.plot(x, hyperparameter_results[1], color='orange',marker='o', label=datasets[1])
    max_y = max(hyperparameter_results[1])
    max_x = x[hyperparameter_results[1].index(max_y)]    
    ax1.scatter([max_x], [max_y], color='red', marker='*', s=100)  # 标记最高点
    
    ax1.plot(x, hyperparameter_results[2],color='green', marker='o', label=datasets[2])
    max_y = max(hyperparameter_results[2])
    max_x = x[hyperparameter_results[2].index(max_y)]    
    ax1.scatter([max_x], [max_y], color='red', marker='*', s=100)  # 标记最高点
    ax1.set_ylim(0.8,1)
    
    plt.subplots_adjust(hspace=0.05)  # 可以调整这个值来控制上下子图的间距

    # 使用spines隐藏上图的底部边界和下图的顶部边界
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # 添加断层指示，如斜线
    d = .015  # 斜线的长度
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左上角斜线
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右上角斜线

    kwargs.update(transform=ax2.transAxes)  # 转换到ax2的坐标系
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左下角斜线
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右下角斜线
    
    ax2.set_xlabel(hyper_names)
    ax2.set_ylabel('AUC')
    ax2.set_xticks(x)
    yticks_ax1 = ax1.get_yticks()
    yticks_ax2 = ax2.get_yticks()
    ax1.set_yticks(yticks_ax1[1:])
    ax2.set_yticks(yticks_ax2[1:])   
    plt.scatter([max_x], [max_y], color='red', marker='*', s=100, label='Highest Point')  
    
    # 图例
    # 收集两个子图的图例句柄和标签
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    if hyper_names == 'rate1(1e-4)':
        ax2.legend(handles, labels, loc='lower right',fontsize='small')
    else: 
        ax2.legend(handles, labels, loc='lower left',fontsize='small')
    
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(bottom=0.2)  # 增加底部边距
    plt.savefig("charts/hyper_"+hyper_names+".png")
    plt.tight_layout()
    plt.show()

# 示例数据
# datasets = ['Clothing', 'Computer', 'Photo']
# num_datasets = len(datasets)
file_path = './hyper/'
files = ['amazon_electronics_computers.csv'] #'Amazon_clothing.csv' , 'amazon_electronics_computers.csv', 'amazon_electronics_photo.csv'
datasets = ['Computer'] # 'Clothing', 'Computer', 'Photo'
epi_list = [5] # ,10,15,20

# 读取数据并画折线图
for file, dataset in zip(files, datasets):
    data = pd.read_csv(file_path + file)

    for epi in epi_list:
        X = [60,80,100,120,140]
        Y = [0.001,0.003,0.005, 0.007, 0.009] 
        Z = [[] for _ in range(len(X))]
        for index, row in data.iterrows():
            match = re.search(f'num_episodes_{int(epi)}_samples_(\d+)_lr_policy_0.005_ano_rate_(\d+.\d+)', row[0]) # \d+.\d+
            if match:
                hyper2_val = int(match.group(1))
                hyper3_val = float(match.group(2))
                performance = row.iloc[-2]
                # X.append(hyper2_val)
                # Y.append(hyper3_val) 
                if hyper3_val in Y: 
                    Z[X.index(hyper2_val)].append(performance*100)
        # 热力图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(X, Y)
        Z = np.array(Z).reshape((len(X), len(Y)))
        # Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')  # 使用面积图表示数据
        ax.set_xticks([60, 80, 100, 120, 140])
        ax.set_yticks([0.001,0.003,0.005, 0.007, 0.009])
        ax.set_zlim(72.00,78.00) 
        ax.set_xlabel(r'Limitation $n_t$')
        ax.set_ylabel(r'Rate $\alpha$')
        ax.set_zlabel('AUC(%)')
        # ax.set_title(f'{dataset}')

        fig.colorbar(surf, shrink=0.5, aspect=10,  location='left', pad=0.01)  # 添加颜色条
        # ax.legend()
        plt.tight_layout()
        plt.savefig(f'./plot/charts/hyper_{dataset}.png')
        plt.close(fig)

# hyperparameter1_results = [[0.562017404,0.653079865, 0.645458268,0.644622325,0.633596509],
#                            [0.94416836, 0.91437176, 0.91694254, 0.89513863, 0.9258454],
#                            [0.919704379,0.897940283,0.910683856,0.914533918,0.918405148]]
# hyperparameter2_results = [[0.664116313,0.653079865,0.646265657,0.64404154],
#                            [0.90362172, 0.92227356, 0.91476629, 0.89532871],
#                            [0.89816942, 0.89476358, 0.90934864, 0.92374023]]
# hyperparameter3_results = [[0.64431644,0.653427061,0.572665345,0.64432282],
#                            [0.90102815, 0.94300064, 0.89219877,  0.91984632],
#                            [0.89164382,0.92473858, 0.91024134, 0.90909011]]

# plot_hyperparameters_analysis(datasets, hyperparameter1_results,x[0],hyper_names[0])
# plot_hyperparameters_analysis(datasets, hyperparameter2_results,x[1],hyper_names[1])
# plot_hyperparameters_analysis(datasets, hyperparameter3_results,x[2],hyper_names[2])