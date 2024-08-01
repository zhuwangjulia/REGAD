import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.interpolate import griddata
import seaborn as sns
# 文件路径
file_path = './hyper/'
files = ['amazon_electronics_photo.csv' ] #'Amazon_clothing.csv' , 'amazon_electronics_computers.csv', 'amazon_electronics_photo.csv'
datasets = [ 'Photo'] # 'Clothing', 'Computer', 'Photo'
epi_list = [5,10,15,20 ] # ,10,15,20
# colors = ['r', 'g', 'b']
# episode_points = {epi: [] for epi in epi_list}

fig, ax = plt.subplots()

for file, dataset in zip(files, datasets):
    data = pd.read_csv(file_path + file)
    for epi in epi_list: 
        hyper1_val = []
        performance = []
        for index, row in data.iterrows():
            match = re.search(f'num_episodes_{epi}_samples_(\d+)_lr_policy_0.005_ano_rate_0.005', row[0])
            if match:
                hyper1_val.append(int(match.group(1)))
                performance.append(row.iloc[-2]*100)
     
        ax.plot(hyper1_val, performance, marker='o', linestyle='-', label=f'T={epi}')
    
    ax.set_xlabel(r'Limitation $n_t$')
    ax.set_ylabel('AUC(%)')
    ax.set_xticks([60,80,100,120,140])
    ax.set_ylim(92.85, 93.1)  
    ax.legend(loc = 'lower left')

# for i, (file, dataset, color) in enumerate(zip(files, datasets, colors)):
#     data = pd.read_csv(file_path + file)
    
#     hyper_val = []
#     performance_bar = []
#     for index, row in data.iterrows():
#         match = re.search(f'num_episodes_5_samples_(\d+)_lr_policy_0.005_ano_rate_0.005', row[0])
#         if match:
#             hyper_val.append(int(match.group(1)) + i*2)
#             performance_bar.append(row.iloc[-2]*100)
#     ax_bar.bar(hyper_val, performance_bar, color=color)

# ax_bar.set_xlabel(r'Limitation $n_t$')
# ax_bar.set_ylabel('AUC(%)')
# ax_bar.set_ylim(62, 94) 

    plt.tight_layout()
    plt.savefig(f'./plot/charts/performance_n_{dataset}.png')
    plt.close(fig)


# 多个episode  
    #     ax.plot(hyper1_val, performance, marker='o', linestyle='-', label=f'T={epi}')
    # ax.set_xlabel(r'Limitation $n_t$')
    # # ax.set_ylabel(r'Rate $\alpha$')
    # ax.set_ylabel('AUC(%)')
    # ax.set_ylim(92.80,93.20) 
    # # ax.set_title(f'Performance of Samples for {dataset}')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(f'./plot/charts/performance_n_{dataset}.png')
    # plt.close(fig)

fig1, ax1 = plt.subplots()
for file, dataset in zip(files, datasets):
    data = pd.read_csv(file_path + file)
    for epi in epi_list: 
        hyper1_val = []
        performance = []
        for index, row in data.iterrows():
            match = re.search(f'num_episodes_{epi}_samples_100_lr_policy_0.005_ano_rate_(\d+.\d+)', row[0])
            if match:
                hyper1_val.append(match.group(1))
                performance.append(row.iloc[-2]*100)
        ax1.plot(hyper1_val, performance, marker='^', linestyle='-', label=f'T={epi}')

    ax1.set_xlabel(r'Rate $\alpha$')
    ax1.set_ylabel('AUC(%)')
    # ax1.set_xticks([0.001,0.003,0.005,0.007,0.009])
    ax1.set_ylim(92.85, 93.1)  

    ax1.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'./plot/charts/performance_a_{dataset}.png')
    plt.close(fig1)

    
# 绘制热力图 
    #     for index, row in data.iterrows():
    #         match = re.search(f'num_episodes_{int(epi)}_samples_(\d+)_lr_policy_0.005_ano_rate_(\d+.\d+)', row[0])
    #         if match:
    #             # hyper1_val = int(match.group(1))
    #             hyper2_val = int(match.group(1))
    #             hyper3_val = float(match.group(2))
    #             performance = row.iloc[-2]
    #             X.append(hyper2_val)
    #             Y.append(hyper3_val) 
    #             Z.append(performance)
    #             if performance >= best_performance:
    #                 best_index = index 
    #                 hyper2_best = hyper2_val 
    #                 hyper3_best = hyper3_val
    #     df = pd.DataFrame({'hyper2_val': X, 'hyper3_val': Y, 'performance': Z})
    #     pivot_table = df.pivot_table(index='hyper3_val', columns='hyper2_val', values='performance')
        
    #     sns.heatmap(pivot_table, ax=axs[i], cmap='viridis', annot=True, fmt=".2f")

    #     axs[i].set_title(f'{dataset} T={epi}')
    #     axs[i].set_xlabel('limitation')
    #     axs[i].set_ylabel('rate')
        
    # plt.tight_layout()
    # plt.savefig(f'./plot/charts/hyper_analysis_{dataset}.png')
    # plt.close(fig) 

    


# #3D曲面图 example
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # Generate data for the surface plot
# X = np.linspace(-5, 5, 100)
# Y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(X, Y)
# Z = np.sin(np.sqrt(X**2 + Y**2))

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface
# surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# # Add labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Surface Plot')

# # Show the plot
# plt.savefig(f'./plot/charts/performance_episodes.png')