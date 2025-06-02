import numpy as np
import pandas as pd

# 初始化训练数据集 [打斗次数, 接吻次数, 类别(-1:爱情片, 1:动作片)]
T = [[3, 104, -1],
     [2, 100, -1],
     [1, 81, -1],
     [101, 10, 1],
     [99, 5, 1],
     [98, 2, 1]
     ]

# 定义待预测的电影和K值
x_test = [46, 50]  # 46次打斗, 50次接吻
K = 5              # 考虑5个最近邻居

# 用于存储每个样本与测试样本的距离及其标签
# 格式: [[距离1, 标签1], [距离2, 标签2], ...]
listdistance = []
for t in T:
    dis = np.sum((np.array(x_test)-np.array(t[:-1]))**2) ** 0.5
    listdistance.append([dis, t[-1]])
print(listdistance)

listdistance.sort()
print(listdistance)

# 选取K个近邻的标签
arr = np.array(listdistance[:K])[:, -1]
print(arr)

a = pd.Series(arr).value_counts()
pre = a.idxmax()
print(a)
print(pre)
