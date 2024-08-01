import matplotlib.pyplot as plt
import numpy as np

# 假设data是一个字典，键为epoch编号，值为每个epoch中各episode的两次cut edges列表
data = {
    'Epoch 1': [(410,8), (545, 4), (475, 3), (702, 9), (637, 6)],
    'Epoch 2': [(531, 5), (793, 8), (595, 11), (649, 15), (700, 9)],
    'Epoch 3': [(420, 3), (503, 4), (496, 9), (523, 5), (301, 1)],
    'Epoch 4': [(169,0),(132,2),(141,1),(131,1), (147,1)],
    'Epoch 5': [(721,1),(565,13),(613,4),(660,12), (744,18)],
}
data_computer = {
    'Epoch 1': [(77,0), (63, 0), (113, 3), (79, 0), (66, 0)],
    'Epoch 2': [(78, 11), (353, 0), (359, 0), (357, 0), (354, 0)],
    'Epoch 3': [(71, 0), (358, 0), (366, 0), (362, 0), (369, 0)],
    'Epoch 4': [(54,0),(55,0),(65,0),(91,0), (101,0)],
    'Epoch 5': [(91,0),(93,0),(110,0),(84,0), (103,0)],
}
data_photo = {
    'Epoch 1': [(313,6), (1000, 0), (994, 0), (971, 0), (980, 0)],
    'Epoch 2': [(52, 0), (191, 0), (148, 0), (146, 0), (104, 0)],
    'Epoch 3': [(317, 2), (983, 0), (1017, 0), (1022, 0), (1039, 0)],
    'Epoch 4': [(62,0),(50,0),(58,0),(51,0), (38,0)],
    'Epoch 5': [(102,0),(104,0),(111,1),(170,0), (173,0)],
}

# 绘制每个epoch的条形图
fig, ax = plt.subplots()
width = 0.4  # 条形的宽度
num_epochs = len(data)  # 计算epoch的数量
index = np.arange(5 * num_epochs) # 五个episodes的横坐标
colors = ['blue', 'green', 'orange', 'purple', 'brown']
xtick_positions = []

for i, (epoch, episodes) in enumerate(data_photo.items()):
    first_cuts = [x[0] for x in episodes]
    second_cuts = [x[1] for x in episodes]
    mean_cuts = np.mean([sum(x) for x in episodes])

    # 计算每个epoch的起始位置
    start_pos = i * 5
    # 计算每个柱形的具体位置
    bar_positions = index[start_pos:start_pos+5] + width * i 

    # 绘制堆叠条形图
    p1 = ax.bar(bar_positions, first_cuts, width, color=colors[i % len(colors)])
    p2 = ax.bar(bar_positions, second_cuts, width, bottom=first_cuts, color='grey')

    # 添加平均线
    line = ax.hlines(mean_cuts, bar_positions[0]-0.5*width, bar_positions[-1] + 0.5*width, colors='red')
    if i == 0:
        # first_cut_patch = p1
        second_cut_patch = p2
        average_line = line
    xtick_positions.append(np.mean(bar_positions))

# 设置图表标题和坐标轴标签

# ax.set_xlabel('Clothing')
ax.set_ylabel('Selected Edges')

ax.legend([second_cut_patch, average_line], ['Second Cut', 'Average'])
ax.set_xticks(xtick_positions)
epoch_labels = [f'Epoch {i+1}' for i in range(num_epochs)]
ax.set_xticklabels(epoch_labels)

plt.tight_layout()
plt.savefig('./plot/charts/edges_photo.png')
plt.close()