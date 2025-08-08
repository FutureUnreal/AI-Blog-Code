import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def J_2d(theta1, theta2):
    """二维目标函数"""
    return 0.6 * (theta1 + theta2)**2 - theta1 * theta2

def dJ_2d(theta1, theta2):
    """二维目标函数的梯度"""
    d_theta1 = 1.2 * theta1 + 0.2 * theta2
    d_theta2 = 0.2 * theta1 + 1.2 * theta2
    return np.array([d_theta1, d_theta2])

# 初始化参数
theta = np.array([6.0, 7.0])
learning_rate = 0.5
max_epochs = 500
epsilon = 1e-10

theta_history = [theta.copy()]
loss_current = J_2d(theta[0], theta[1])
loss_change = loss_current
iter_num = 0

print(f"初始值: theta = ({theta[0]:.5f}, {theta[1]:.5f}), J(theta) = {loss_current:.5f}")

# 梯度下降迭代
while loss_change > epsilon and iter_num < max_epochs:
    iter_num += 1
    gradient = dJ_2d(theta[0], theta[1])
    theta = theta - learning_rate * gradient

    loss_new = J_2d(theta[0], theta[1])
    loss_change = np.abs(loss_current - loss_new)
    loss_current = loss_new

    theta_history.append(theta.copy())

    # 每50次迭代打印一次结果
    if iter_num % 50 == 0:
        print(f"第{iter_num}次迭代: theta = ({theta[0]:.5f}, {theta[1]:.5f}), J(theta) = {loss_current:.5f}")

theta_history = np.array(theta_history)
print(f"\n最终结果: theta = ({theta[0]:.5f}, {theta[1]:.5f}), J(theta) = {loss_current:.5f}")
print(f"迭代次数: {iter_num}")

# 可视化结果
# 创建网格
theta1_range = np.linspace(-8, 8, 100)
theta2_range = np.linspace(-8, 8, 100)
Theta1, Theta2 = np.meshgrid(theta1_range, theta2_range)
Z = J_2d(Theta1, Theta2)

# 创建3D图
fig = plt.figure(figsize=(15, 5))

# 3D曲面图
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(Theta1, Theta2, Z, alpha=0.6, cmap='viridis')
ax1.plot(theta_history[:, 0], theta_history[:, 1],
         [J_2d(t[0], t[1]) for t in theta_history],
         'ro-', linewidth=2, markersize=3, label='梯度下降路径')
ax1.set_xlabel('θ1')
ax1.set_ylabel('θ2')
ax1.set_zlabel('J(θ1, θ2)')
ax1.set_title('3D曲面图')

# 等高线图
ax2 = fig.add_subplot(132)
contour = ax2.contour(Theta1, Theta2, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(theta_history[:, 0], theta_history[:, 1], 'ro-', linewidth=2, markersize=3)
ax2.plot(theta_history[0, 0], theta_history[0, 1], 'go', markersize=8, label='起始点')
ax2.plot(theta_history[-1, 0], theta_history[-1, 1], 'ro', markersize=8, label='终点')
ax2.set_xlabel('θ1')
ax2.set_ylabel('θ2')
ax2.set_title('等高线图')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 损失函数变化
ax3 = fig.add_subplot(133)
losses = [J_2d(t[0], t[1]) for t in theta_history]
ax3.plot(losses, 'b-', linewidth=2)
ax3.set_xlabel('迭代次数')
ax3.set_ylabel('损失值')
ax3.set_title('损失函数变化')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 测试不同起始点的影响
starting_points = [
    [6.0, 7.0],
    [-5.0, 3.0],
    [2.0, -4.0],
    [-3.0, -2.0]
]

plt.figure(figsize=(12, 8))

# 创建等高线背景
contour = plt.contour(Theta1, Theta2, Z, levels=15, alpha=0.6)
plt.clabel(contour, inline=True, fontsize=8)

colors = ['red', 'blue', 'green', 'orange']

for i, start_point in enumerate(starting_points):
    # 运行梯度下降
    theta_test = np.array(start_point)
    learning_rate_test = 0.5
    max_epochs_test = 200
    epsilon_test = 1e-10
    theta_history_test = [theta_test.copy()]
    loss_current_test = J_2d(theta_test[0], theta_test[1])
    loss_change_test = loss_current_test
    iter_num_test = 0

    while loss_change_test > epsilon_test and iter_num_test < max_epochs_test:
        iter_num_test += 1
        gradient = dJ_2d(theta_test[0], theta_test[1])
        theta_test = theta_test - learning_rate_test * gradient

        loss_new = J_2d(theta_test[0], theta_test[1])
        loss_change_test = np.abs(loss_current_test - loss_new)
        loss_current_test = loss_new

        theta_history_test.append(theta_test.copy())

    theta_history_test = np.array(theta_history_test)

    # 绘制路径
    plt.plot(theta_history_test[:, 0], theta_history_test[:, 1],
            color=colors[i], linewidth=2, marker='o', markersize=2,
            label=f'起始点 ({start_point[0]}, {start_point[1]})')
    plt.plot(start_point[0], start_point[1], 'o', color=colors[i], markersize=8)

plt.xlabel('θ1')
plt.ylabel('θ2')
plt.title('不同起始点的梯度下降路径')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
