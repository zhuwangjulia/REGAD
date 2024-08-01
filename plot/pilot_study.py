import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# 模型和数据集名称
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
models = ['DeepSAD', 'BWGNN', 'Meta-GDN', 'GHRN']
datasets = ['Clothing', 'Computer', 'Photo']

# 假设每个模型都有三个数据集的前后对比结果
# 结构为 [模型][数据集][前后]
auc_values = [
    [[0.7949, 0.5585], [0.7153, 0.5331], [0.9725, 0.9118]],  # DeepSAD 的三个数据集的前后对比
    [[0.7982, 0.5118], [0.9775, 0.5009], [0.8094, 0.5012]],  # Model 2
    [[0.8433, 0.5985], [0.9942, 0.6055], [0.9525, 0.8764]],  # Model 3
    [[0.7979, 0.5127], [0.9612, 0.4994], [0.8159, 0.5023]]   # Model 4
]

num_models = len(models)
num_datasets = len(datasets)
colors = ['blue', 'green', 'orange']

# 设置条形图的位置和宽度
bar_width = 0.2
index = np.arange(num_models)  # 模型数量决定了主要的x轴刻度

# # 绘制条形图
# for i, model in enumerate(models):
#     for j, dataset in enumerate(datasets):
#         pos = i * num_datasets + j
#         before_injection = auc_values[i][j][0]
#         after_injection = auc_values[i][j][1]

#         plt.bar(pos, before_injection, bar_width, color=colors[j], alpha=0.5,
#                 label='Before Injection' if i == 0 and j == 0 else "")
#         plt.bar(pos, after_injection, bar_width, color=colors[j],
#                 label='After Injection' if i == 0 and j == 0 else "")
fig, ax = plt.subplots()

# 模型在x轴上的基础位置
model_positions = np.arange(len(models))

# 数据集在每个模型内的偏移量
dataset_offsets = np.linspace(-0.2, 0.2, num=len(datasets))

# 绘制散点图并添加箭头
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        before, after = auc_values[i][j]
        # 为每个数据集计算独特的x轴位置
        x_pos = model_positions[i] + dataset_offsets[j]

        ax.scatter([x_pos], [before], color=colors[j], edgecolor='black',
                   label=f'{dataset} Before' if i == 0 else "")
        ax.scatter([x_pos], [after], color=colors[j], alpha=0.5,
                   edgecolor='black', label=f'{dataset} After' if i == 0 else "")
        # 添加箭头
        ax.annotate("", xy=(x_pos, after), xytext=(x_pos, before),
                    arrowprops=dict(arrowstyle="->", color=colors[j]))
# 自定义图例
legend_elements = [Line2D([0], [0], marker='o', color='w', label=dataset,
                          markerfacecolor=color, markersize=10) for dataset, color in zip(datasets, colors)]
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Before',
                              markerfacecolor='w', markeredgecolor='black', markersize=10))
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='After',
                              markerfacecolor='w', markeredgecolor='black', alpha=0.5, markersize=10))

ax.legend(handles=legend_elements, loc='lower right')

# 设置横坐标
plt.xticks(model_positions, models)

# 设置纵坐标的显示范围
plt.ylim(0.2, 1.1)

# 添加标题和轴标签
# plt.title('AUC Comparison Across Models and Datasets')
plt.xlabel('GAD Models')
plt.ylabel('AUC')


if not os.path.exists('charts'):
    os.makedirs('charts')

# 保存图形到文件
plt.savefig('charts/auc_comparison.png', dpi=300)  # dpi参数控制图像的分辨率
# 显示图形
plt.tight_layout()
plt.show()
