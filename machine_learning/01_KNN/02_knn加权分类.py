import numpy as np


T = [[3, 104, -1],
     [2, 100, -1],
     [1, 81, -1],
     [101, 10, 1],
     [99, 5, 1],
     [98, 2, 1]
     ]

# 定义待预测的电影和K值
x_test = [50, 50]
K = 3

listdistance = []

for t in T:
    dis = np.sum((np.array(x_test)-np.array(t[:-1]))**2) ** 0.5
    listdistance.append([dis, t[-1]])
print(listdistance)

listdistance.sort()
print(listdistance)

# 取前K个近邻并计算权重（距离越近权重越大）
weight = [1 / (i[0] + 0.001) for i in listdistance[:K]]  # 加0.001防止除零错误

# 归一化权重
weight_sum = sum(weight)
normalized_weight = [w / weight_sum for w in weight]

# 加权投票预测（二分类情况）
weighted_vote = sum([normalized_weight[i] * listdistance[:K][i][1] for i in range(K)])
prediction = 1 if weighted_vote > 0 else -1

print(prediction)
