import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

def train_func(X, K):
    """训练函数：y = K * x"""
    return K * X

# 准备数据
EXAMPLE_NUM = 100
BATCH_SIZE = 10
TRAIN_STEPS = 150
LEARNING_RATE = 0.0001

# 生成训练数据
X_INPUT = np.arange(EXAMPLE_NUM) * 0.1
Y_OUTPUT_CORRECT = 5 * X_INPUT  # 真实函数 y = 5x

print(f"数据集大小: {EXAMPLE_NUM}")
print(f"批量大小: {BATCH_SIZE}")
print(f"训练步数: {TRAIN_STEPS}")
print(f"学习率: {LEARNING_RATE}")
print(f"真实参数: K = 5")
print("-" * 40)

# BGD 实现（按照参考代码的方式，不除以样本数）
k_BGD = 0.0
k_BGD_RECORD = [k_BGD]
for step in range(TRAIN_STEPS):
    SUM_BGD = 0
    for index in range(len(X_INPUT)):
        SUM_BGD += (train_func(X_INPUT[index], k_BGD) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    k_BGD -= LEARNING_RATE * SUM_BGD  # 注意：这里没有除以样本数
    k_BGD_RECORD.append(k_BGD)

print(f"BGD 最终参数: K = {k_BGD:.5f}")

# SGD 实现
np.random.seed(42)
k_SGD = 0.0
k_SGD_RECORD = [k_SGD]
for step in range(TRAIN_STEPS):
    index = np.random.randint(len(X_INPUT))
    SUM_SGD = (train_func(X_INPUT[index], k_SGD) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    k_SGD -= LEARNING_RATE * SUM_SGD
    k_SGD_RECORD.append(k_SGD)

print(f"SGD 最终参数: K = {k_SGD:.5f}")

# MBGD 实现
np.random.seed(42)
k_MBGD = 0.0
k_MBGD_RECORD = [k_MBGD]
for step in range(TRAIN_STEPS):
    SUM_MBGD = 0
    index_start = np.random.randint(len(X_INPUT) - BATCH_SIZE)
    for index in np.arange(index_start, index_start + BATCH_SIZE):
        SUM_MBGD += (train_func(X_INPUT[index], k_MBGD) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    k_MBGD -= LEARNING_RATE * SUM_MBGD
    k_MBGD_RECORD.append(k_MBGD)

print(f"MBGD 最终参数: K = {k_MBGD:.5f}")

# 可视化比较结果
plt.figure(figsize=(12, 8))

# 主图：参数收敛过程
plt.subplot(2, 2, 1)
steps = np.arange(TRAIN_STEPS + 1)
plt.plot(steps, k_BGD_RECORD, 'r-', linewidth=2, label='BGD')
plt.plot(steps, k_SGD_RECORD, 'g-', linewidth=2, label='SGD')
plt.plot(steps, k_MBGD_RECORD, 'b-', linewidth=2, label='MBGD')
plt.axhline(y=5, color='purple', linestyle='--', linewidth=2, label='真实值 (K=5)')
plt.xlabel('训练步数')
plt.ylabel('参数 K')
plt.title('三种梯度下降方法的收敛过程')
plt.legend()
plt.grid(True, alpha=0.3)

# 局部放大图（最后50步）
plt.subplot(2, 2, 2)
start_idx = max(0, TRAIN_STEPS - 50)
steps_zoom = steps[start_idx:]
plt.plot(steps_zoom, k_BGD_RECORD[start_idx:], 'r-', linewidth=2, label='BGD')
plt.plot(steps_zoom, k_SGD_RECORD[start_idx:], 'g-', linewidth=2, label='SGD')
plt.plot(steps_zoom, k_MBGD_RECORD[start_idx:], 'b-', linewidth=2, label='MBGD')
plt.axhline(y=5, color='purple', linestyle='--', linewidth=2, label='真实值')
plt.xlabel('训练步数')
plt.ylabel('参数 K')
plt.title('收敛过程（最后50步）')
plt.legend()
plt.grid(True, alpha=0.3)

# 误差分析
plt.subplot(2, 2, 3)
bgd_errors = np.abs(np.array(k_BGD_RECORD) - 5)
sgd_errors = np.abs(np.array(k_SGD_RECORD) - 5)
mbgd_errors = np.abs(np.array(k_MBGD_RECORD) - 5)

plt.semilogy(steps, bgd_errors, 'r-', linewidth=2, label='BGD')
plt.semilogy(steps, sgd_errors, 'g-', linewidth=2, label='SGD')
plt.semilogy(steps, mbgd_errors, 'b-', linewidth=2, label='MBGD')
plt.xlabel('训练步数')
plt.ylabel('绝对误差 (对数尺度)')
plt.title('参数误差变化')
plt.legend()
plt.grid(True, alpha=0.3)

# 方差分析
plt.subplot(2, 2, 4)
window_size = 10
bgd_variance = [np.var(k_BGD_RECORD[max(0, i-window_size):i+1]) for i in range(len(k_BGD_RECORD))]
sgd_variance = [np.var(k_SGD_RECORD[max(0, i-window_size):i+1]) for i in range(len(k_SGD_RECORD))]
mbgd_variance = [np.var(k_MBGD_RECORD[max(0, i-window_size):i+1]) for i in range(len(k_MBGD_RECORD))]

plt.plot(steps, bgd_variance, 'r-', linewidth=2, label='BGD')
plt.plot(steps, sgd_variance, 'g-', linewidth=2, label='SGD')
plt.plot(steps, mbgd_variance, 'b-', linewidth=2, label='MBGD')
plt.xlabel('训练步数')
plt.ylabel('方差')
plt.title('参数更新的方差')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印最终对比结果
print("\n" + "=" * 50)
print("最终结果对比")
print("=" * 50)
print(f"真实值: K = 5.0")
print(f"BGD ({TRAIN_STEPS}步): K = {k_BGD:.5f}, 误差 = {abs(k_BGD - 5):.5f}")
print(f"SGD ({TRAIN_STEPS}步): K = {k_SGD:.5f}, 误差 = {abs(k_SGD - 5):.5f}")
print(f"MBGD ({TRAIN_STEPS}步): K = {k_MBGD:.5f}, 误差 = {abs(k_MBGD - 5):.5f}")

print(f"\n方差对比（最后10步）:")
print(f"BGD方差: {np.var(k_BGD_RECORD[-10:]):.8f}")
print(f"SGD方差: {np.var(k_SGD_RECORD[-10:]):.8f}")
print(f"MBGD方差: {np.var(k_MBGD_RECORD[-10:]):.8f}")
