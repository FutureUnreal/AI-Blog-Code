import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载波士顿房价数据集
data = pd.read_csv('datas/boston_housing.data', sep='\s+', header=None)

# 提取特征和目标变量
X = data.iloc[:, :-1]  # 所有行，除了最后一列
y = data.iloc[:, -1]   # 所有行，只取最后一列（房价）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===============================
# 1.不使用多项式扩展的基准模型
# ===============================

# 创建并训练线性回归模型
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)

# 预测并评估
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# 计算R²分数
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"不使用多项式扩展：")
print(f"训练集R²: {train_r2:.2f}")
print(f"测试集R²: {test_r2:.2f}")

# ===============================
# 2.使用二项式扩展
# ===============================

# 创建二阶多项式特征转换器
poly = PolynomialFeatures(degree=2, include_bias=False)

# 对训练集进行特征转换
X_train_poly = poly.fit_transform(X_train)
# 对测试集进行特征转换（注意只使用transform，不使用fit_transform）
X_test_poly = poly.transform(X_test)

print(f"原始特征数量: {X_train.shape[1]}")
print(f"多项式扩展后特征数量: {X_train_poly.shape[1]}")

# 创建并训练线性回归模型
lr_poly = LinearRegression(fit_intercept=True)
lr_poly.fit(X_train_poly, y_train)

# 预测并评估
y_train_poly_pred = lr_poly.predict(X_train_poly)
y_test_poly_pred = lr_poly.predict(X_test_poly)

# 计算R²分数
train_poly_r2 = r2_score(y_train, y_train_poly_pred)
test_poly_r2 = r2_score(y_test, y_test_poly_pred)

print(f"\n使用二阶多项式扩展：")
print(f"训练集R²: {train_poly_r2:.2f}")
print(f"测试集R²: {test_poly_r2:.2f}")

# ===============================
# 3.使用三项式扩展
# ===============================

# 创建三阶多项式特征转换器
poly3 = PolynomialFeatures(degree=3, include_bias=False)

# 对训练集进行特征转换
X_train_poly3 = poly3.fit_transform(X_train)
# 对测试集进行特征转换
X_test_poly3 = poly3.transform(X_test)

print(f"\n三阶多项式扩展后特征数量: {X_train_poly3.shape[1]}")

# 创建并训练线性回归模型
lr_poly3 = LinearRegression(fit_intercept=True)
lr_poly3.fit(X_train_poly3, y_train)

# 预测并评估
y_train_poly3_pred = lr_poly3.predict(X_train_poly3)
y_test_poly3_pred = lr_poly3.predict(X_test_poly3)

# 计算R²分数
train_poly3_r2 = r2_score(y_train, y_train_poly3_pred)
test_poly3_r2 = r2_score(y_test, y_test_poly3_pred)

print(f"使用三阶多项式扩展：")
print(f"训练集R²: {train_poly3_r2:.2f}")
print(f"测试集R²: {test_poly3_r2:.2f}")

# ===============================
# 4.只保留交互项
# ===============================

# 创建只保留交互项的多项式特征转换器
poly_interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# 对训练集进行特征转换
X_train_poly_inter = poly_interaction.fit_transform(X_train)
# 对测试集进行特征转换
X_test_poly_inter = poly_interaction.transform(X_test)

print(f"\n只保留交互项的多项式扩展后特征数量: {X_train_poly_inter.shape[1]}")

# 创建并训练线性回归模型
lr_poly_inter = LinearRegression(fit_intercept=True)
lr_poly_inter.fit(X_train_poly_inter, y_train)

# 预测并评估
y_train_poly_inter_pred = lr_poly_inter.predict(X_train_poly_inter)
y_test_poly_inter_pred = lr_poly_inter.predict(X_test_poly_inter)

# 计算R²分数
train_poly_inter_r2 = r2_score(y_train, y_train_poly_inter_pred)
test_poly_inter_r2 = r2_score(y_test, y_test_poly_inter_pred)

print(f"只保留交互项：")
print(f"训练集R²: {train_poly_inter_r2:.2f}")
print(f"测试集R²: {test_poly_inter_r2:.2f}")

# 1. 线性模型
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_train)), y_train_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('线性模型 - 训练集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

# 测试集
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_test)), y_test_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('线性模型 - 测试集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 2. 二阶多项式
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_train)), y_train_poly_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('二阶多项式 - 训练集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

# 测试集
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_test)), y_test_poly_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('二阶多项式 - 测试集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 3. 三阶多项式
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_train)), y_train_poly3_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('三阶多项式 - 训练集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

# 测试集
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_test)), y_test_poly3_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('三阶多项式 - 测试集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 4. 只保留交互项
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_train)), y_train_poly_inter_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('只保留交互项 - 训练集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

# 测试集
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test, 'b-', linewidth=1.5, label='真实值')
plt.plot(range(len(y_test)), y_test_poly_inter_pred, 'r-', linewidth=1.5, label='预测值')
plt.title('只保留交互项 - 测试集')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()