import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error,r2_score
## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

flag = True
# 一、构造数据
X1 = np.array([
    [10, 1],
    [15, 1],
    [20, 1],
    [30, 1],
    [50, 2],
    [60, 1],
    [60, 2],
    [70, 2]]).reshape((-1, 2))
Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))

if flag:
    # 添加一个截距项对应的X值
    X = np.column_stack((X1, np.ones(shape=(X1.shape[0], 1))))
else:
    # 不加入截距项
    X = X1
print(X)

# 二、根据解析式的公式求解theta的值
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
print(theta)

# 三、 根据求解出来的theta求出预测值
predict_y = X.dot(theta)
# 查看MSE和R^2
print(Y.shape)
print(predict_y.shape)
mse = mean_squared_error(y_true=Y,y_pred=predict_y)
print("MSE",mse)
r2 = r2_score(y_true=Y,y_pred=predict_y)
print("r^2",r2)

# 基于训练好的模型参数对一个未知的样本做一个预测
if flag:
    x = np.array([[55.0, 2.0,1.0]])
else:
    x = np.array([[55.0, 2.0]])
pred_y = x.dot(theta)
print("当面积为55平并且房间数目为2的时候，预测价格为:{}".format(pred_y))


# 四、可视化
x1 = X[:, 0]
x2 = X[:, 1]

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, Y.flatten(), s=40, c='r', depthshade=False)  ###画点图


x1 = np.arange(0, 100)
x2 = np.arange(0, 4)
x1, x2 = np.meshgrid(x1, x2)

def predict(x1, x2, theta, base=False):
    if base:
        y_ = x1 * theta[0] + x2 * theta[1] + theta[2]
    else:
        y_ = x1 * theta[0] + x2 * theta[1]
    return y_
z = np.array(list(map(lambda t: predict(t[0], t[1], theta, base=flag), zip(x1.flatten(), x2.flatten()))))
z.shape = x1.shape
print(z.shape)
ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap='jet')
ax.set_title(u'房屋租赁价格预测')
plt.show()
