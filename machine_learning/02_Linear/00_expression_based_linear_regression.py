# ===============================
# 步骤1：导入必要的库
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, r2_score

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# ===============================
# 步骤2：准备训练数据
# ===============================
# 构造特征矩阵：房屋面积和房间数量
X_train = np.array([
    [10, 1],  # 10平米，1个房间
    [15, 1],  # 15平米，1个房间
    [20, 1],  # 20平米，1个房间
    [30, 1],  # 30平米，1个房间
    [50, 2],  # 50平米，2个房间
    [60, 1],  # 60平米，1个房间
    [60, 2],  # 60平米，2个房间
    [70, 2]   # 70平米，2个房间
]).reshape((-1, 2))

# 构造目标变量：房屋租赁价格（单位：千元）
y_train = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))

# 打印数据形状
print("训练数据：")
print(f"特征矩阵X_train形状: {X_train.shape}")  # 8行2列
print(f"目标变量y_train形状: {y_train.shape}")  # 8行1列

# ===============================
# 步骤3：添加截距项（偏置项）
# ===============================
# 是否使用截距项的标志
use_intercept = True

if use_intercept:
    # 添加截距项：为特征矩阵添加一列全为1的值
    # column_stack函数将两个数组按列合并
    X = np.column_stack((X_train, np.ones(shape=(X_train.shape[0], 1))))
    print("\n添加截距项后的特征矩阵X：")
    print(f"形状: {X.shape}")  # 8行3列（原有2列特征 + 1列截距项）
    print(X)
else:
    # 不使用截距项
    X = X_train
    print("\n不使用截距项的特征矩阵X：")
    print(f"形状: {X.shape}")  # 8行2列
    print(X)

# ===============================
# 步骤4：使用解析式计算最优参数θ
# ===============================
# 参数解析式: θ = (X^T X)^{-1} X^T y

# 第一步：计算X^T X
XTX = X.T.dot(X)
print("\nX^T X矩阵：")
print(XTX)

# 第二步：计算(X^T X)^{-1}
XTX_inv = np.linalg.inv(XTX)
print("\n(X^T X)^{-1}矩阵：")
print(XTX_inv)

# 第三步：计算X^T y
XTy = X.T.dot(y_train)
print("\nX^T y矩阵：")
print(XTy)

# 第四步：计算θ = (X^T X)^{-1} X^T y
theta = XTX_inv.dot(XTy)
print("\n最优参数θ：")
print(theta)

# 如果使用截距项，解释各个参数的含义
if use_intercept:
    print(f"\nθ_0 (面积系数): {theta[0][0]:.6f}")
    print(f"θ_1 (房间数系数): {theta[1][0]:.6f}")
    print(f"θ_2 (截距项): {theta[2][0]:.6f}")

    # 线性回归方程: y = θ_0 * x_1 + θ_1 * x_2 + θ_2
    print(f"\n线性回归方程: y = {theta[0][0]:.4f} * 面积 + {theta[1][0]:.4f} * 房间数 + {theta[2][0]:.4f}")
else:
    print(f"\nθ_0 (面积系数): {theta[0][0]:.6f}")
    print(f"θ_1 (房间数系数): {theta[1][0]:.6f}")

    # 线性回归方程: y = θ_0 * x_1 + θ_1 * x_2
    print(f"\n线性回归方程: y = {theta[0][0]:.4f} * 面积 + {theta[1][0]:.4f} * 房间数")


# ===============================
# 步骤5：模型预测与评估
# ===============================
# 使用训练好的模型进行预测
y_pred = X.dot(theta)

# 计算模型评估指标
mse = mean_squared_error(y_true=y_train, y_pred=y_pred)
r2 = r2_score(y_true=y_train, y_pred=y_pred)

print("\n模型评估：")
print(f"均方误差(MSE): {mse:.6f}")
print(f"决定系数(R^2): {r2:.6f}")

# 预测新样本：55平米、2个房间的房子价格
if use_intercept:
    # 新样本需要添加截距项
    x_new = np.array([[55.0, 2.0, 1.0]])
else:
    # 不使用截距项
    x_new = np.array([[55.0, 2.0]])

# 预测
pred_price = x_new.dot(theta)
print(f"\n预测结果：")
print(f"当面积为55平米并且房间数为2的时候，预测价格为: {pred_price[0][0]:.2f}千元")

# ===============================
# 步骤6：可视化结果
# ===============================
# 提取特征
x1 = X[:, 0]  # 房屋面积
x2 = X[:, 1]  # 房间数量

# 创建3D图
fig = plt.figure(figsize=(10, 7), facecolor='w')
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点（红色）
ax.scatter(x1, x2, y_train.flatten(), s=50, c='r', marker='o',
           depthshade=False, label='实际值')

# 绘制预测数据点（蓝色）
ax.scatter(x1, x2, y_pred.flatten(), s=50, c='b', marker='x',
           depthshade=False, label='预测值')

# 生成网格数据用于绘制预测平面
x1_grid = np.linspace(0, 80, 20)  # 面积范围
x2_grid = np.linspace(0, 3, 10)  # 房间数范围
x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)


# 定义预测函数
def predict_price(x1, x2, theta, has_intercept=False):
    """
    根据学习到的参数预测房价

    参数:
    x1: 房屋面积
    x2: 房间数量
    theta: 模型参数
    has_intercept: 是否使用截距项

    返回:
    预测的房屋价格
    """
    if has_intercept:
        # 有截距项: y = θ₀ * 面积 + θ₁ * 房间数 + θ₂
        return x1 * theta[0][0] + x2 * theta[1][0] + theta[2][0]
    else:
        # 无截距项: y = θ₀ * 面积 + θ₁ * 房间数
        return x1 * theta[0][0] + x2 * theta[1][0]


# 计算平面上每个点的预测值
z_grid = np.zeros_like(x1_grid)
for i in range(x1_grid.shape[0]):
    for j in range(x1_grid.shape[1]):
        z_grid[i, j] = predict_price(x1_grid[i, j], x2_grid[i, j],
                                     theta, has_intercept=use_intercept)

# 绘制预测平面
surf = ax.plot_surface(x1_grid, x2_grid, z_grid,
                       rstride=1, cstride=1, alpha=0.7,
                       cmap='jet', edgecolor='none')

# 添加颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('预测价格（千元）')

# 设置图表标题和标签
ax.set_title('房屋租赁价格预测模型', fontsize=15)
ax.set_xlabel('面积（平方米）', fontsize=12)
ax.set_ylabel('房间数量', fontsize=12)
ax.set_zlabel('价格（千元）', fontsize=12)
ax.legend()

# 调整视角
ax.view_init(elev=10, azim=-70)

plt.tight_layout()
plt.show()