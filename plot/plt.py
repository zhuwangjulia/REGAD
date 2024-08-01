import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
fig, (ax_1,ax_2) =plt.subplots(1, 2, figsize=(7.5, 3.5))

x = [90,70,50,30,10]
y1 = [i * 100 for i in [0.672074626,0.674914209,0.677785888,0.691094751,0.691246782]]
y2 = [i * 100 for i in [0.739001024,0.747903675,0.754460878,0.758936761,0.760825258]]
y3 = [i * 100 for i in [0.904536987,0.917394342,0.930075467,0.931707046,0.925266461]]

ax_1.plot(x, y1, color='orange', marker='o', label='Clothing')
ax_1.plot(x, y2, color='green', marker='x', label='Computer')
ax_1.plot(x, y3, color='purple', marker='^', label='Photo')

ax_1.set_xticks(x)
ax_1.set_ylim(55, 95) 
ax_1.set_xlabel('Noisy label ratio(%)')
ax_1.set_ylabel('AUC(%)')
ax_1.set_title('(a)',loc='center', y=-0.35)
ax_1.legend(loc='lower left', fontsize = 'x-small')

y1b = [i * 100 for i in [0.0903606775,0.093863394,0.092242562,0.102553065,0.097184432]]
y2b = [i * 100 for i in [0.314956276,0.341477319,0.356870959,0.340339369,0.363177862]]
y3b = [i * 100 for i in [0.4697203,0.501693832,0.486198768,0.474142692,0.513236398]]


ax_2.plot(x, y1b, color='orange', marker='o', label='Clothing')
ax_2.plot(x, y2b, color='green', marker='x', label='Computer')
ax_2.plot(x, y3b, color='purple', marker='^', label='Photo')

ax_2.set_xticks(x)
ax_2.set_ylim(0, 69) 
ax_2.set_xlabel('Noisy label ratio(%)')
ax_2.set_ylabel('AUPR(%)')
ax_2.set_title('(b)',loc='center', y=-0.35)
ax_2.legend(loc='upper right', fontsize = 'x-small')

plt.tight_layout()
plt.savefig('./plot/charts/different_rate.png')
plt.close()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5), sharex=True)

x2 = [10,20,30,40]
z1 = [i * 100 for i in [0.670140749,0.677209507,0.677785888,0.679634333]]
z2 = [i * 100 for i in [0.739365536,0.748049919,0.754460878,0.758498027]]
z3 = [i * 100 for i in [0.92121662,0.927596609,0.930075467,0.931877278]]

ax1.plot(x2, z1, color='orange', marker='o', label='Clothing')
ax1.plot(x2, z2, color='green', marker='x', label='Computer')
ax1.plot(x2, z3, color='purple', marker='^', label='Photo')
ax1.set_xlabel('Anomaly nodes number')
ax1.set_ylabel('AUC(%)')
ax1.set_title('(a)',loc='center', y=-0.35)
ax1.legend(loc='lower left', fontsize = 'x-small')

# 假设这是第二个评估标准的数据
z1b = [i * 100 for i in [0.093667049,0.09227065,0.092212562,0.090146687]]
z2b = [i * 100 for i in [0.325623294,0.333635297,0.356870959,0.358489324]]
z3b = [i * 100 for i in [0.500867945,0.489301316,0.486198768,0.474207995]]

ax2.plot(x2, z1b, color='orange', marker='o', label='Clothing')
ax2.plot(x2, z2b, color='green', marker='x', label='Computer')
ax2.plot(x2, z3b, color='purple', marker='^', label='Photo')
ax2.set_xlabel('Anomaly nodes number')
ax2.set_ylabel('AUPR(%)')
ax2.legend(loc='upper right',  fontsize = 'x-small')
ax2.set_title('(b)',loc='center', y=-0.35)

ax1.set_xticks(x2)
ax1.set_ylim(55, 99) 

ax2.set_xticks(x2)
ax2.set_ylim(0, 65) 

# fig.suptitle('Figure Bottom Title', y=0.05)  # y参数控制标题的垂直位置
plt.tight_layout()
plt.savefig('./plot/charts/anomaly_number.png')
plt.close()