import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def J_1d(theta):
    """目标函数"""
    return 0.5 * (theta - 0.25)**2

def dJ_1d(theta):
    """目标函数的导数"""
    return theta - 0.25

# 初始化参数
theta = 8.0  # 初始点
learning_rate = 0.5
max_epochs = 150
epsilon = 1e-10  # 容忍的误差范围

theta_history = [theta]
loss_current = J_1d(theta)
loss_change = loss_current
iter_num = 0

print(f"初始值: theta = {theta:.5f}, J(theta) = {loss_current:.5f}")

# 梯度下降迭代
while loss_change > epsilon and iter_num < max_epochs:
    iter_num += 1
    gradient = dJ_1d(theta)
    theta = theta - learning_rate * gradient

    loss_new = J_1d(theta)
    loss_change = np.abs(loss_current - loss_new)
    loss_current = loss_new

    theta_history.append(theta)

    # 每10次迭代打印一次结果
    if iter_num % 10 == 0:
        print(f"第{iter_num}次迭代: theta = {theta:.5f}, J(theta) = {loss_current:.5f}")

print(f"\n最终结果: theta = {theta:.5f}, J(theta) = {loss_current:.5f}")
print(f"迭代次数: {iter_num}")

# 可视化结果
x_plot = np.arange(-8, 8.5, 0.01)
y_plot = J_1d(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='函数 $J(θ) = 0.5 * (θ - 0.25)^2$')
plt.plot(theta_history, J_1d(np.array(theta_history)), 'bo--', linewidth=2,
         markersize=4, label='梯度下降路径')

# 标记起始点和终点
plt.plot(theta_history[0], J_1d(theta_history[0]), 'go', markersize=8, label='起始点')
plt.plot(theta_history[-1], J_1d(theta_history[-1]), 'ro', markersize=8, label='终点')

plt.title(f'一维梯度下降\n学习率: 0.5, 最终解: ({theta_history[-1]:.3f}, {loss_current:.3f}), 迭代次数: {iter_num}')
plt.xlabel('θ')
plt.ylabel('J(θ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 测试不同学习率的影响
learning_rates = [0.1, 0.5, 0.9, 1.1]

plt.figure(figsize=(15, 10))

for i, lr in enumerate(learning_rates):
    plt.subplot(2, 2, i+1)

    # 重新运行梯度下降
    theta_test = 8.0
    max_epochs_test = 50
    theta_history_test = [theta_test]
    loss_current_test = J_1d(theta_test)
    loss_change_test = loss_current_test
    iter_num_test = 0

    while loss_change_test > epsilon and iter_num_test < max_epochs_test:
        iter_num_test += 1
        gradient = dJ_1d(theta_test)
        theta_test = theta_test - lr * gradient

        loss_new = J_1d(theta_test)
        loss_change_test = np.abs(loss_current_test - loss_new)
        loss_current_test = loss_new

        theta_history_test.append(theta_test)

    # 绘图
    plt.plot(x_plot, y_plot, 'r-', linewidth=2)
    plt.plot(theta_history_test, J_1d(np.array(theta_history_test)), 'bo--', linewidth=2, markersize=3)
    plt.title(f'学习率 = {lr}, 迭代次数 = {iter_num_test}')
    plt.xlabel('θ')
    plt.ylabel('J(θ)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
