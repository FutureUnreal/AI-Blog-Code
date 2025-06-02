import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('datas/boston_housing.data', sep='\s+', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

lr_poly3 = LinearRegression(fit_intercept=True)
lr_poly3.fit(X_train_poly3, y_train)

# 获取参数
coefficients = lr_poly3.coef_
intercept = lr_poly3.intercept_

print(f"参数总数：{len(coefficients)}")
print(f"截距项: {intercept:.4f}")
print(f"最大参数值: {np.max(coefficients):.4f}")
print(f"最小参数值: {np.min(coefficients):.4f}")
print(f"参数均值: {np.mean(coefficients):.4f}")
print(f"参数标准差: {np.std(coefficients):.4f}")

# 按数量级统计参数
magnitude_bins = {
    "10-100": 0,
    "1-10": 0,
    "0.1-1": 0,
    "0.01-0.1": 0,
    "0.001-0.01": 0,
    "<0.001": 0
}

for coef in np.abs(coefficients):
    if coef > 10:
        magnitude_bins["10-100"] += 1
    elif coef > 1:
        magnitude_bins["1-10"] += 1
    elif coef > 0.1:
        magnitude_bins["0.1-1"] += 1
    elif coef > 0.01:
        magnitude_bins["0.01-0.1"] += 1
    elif coef > 0.001:
        magnitude_bins["0.001-0.01"] += 1
    else:
        magnitude_bins["<0.001"] += 1

print("\n参数数量级分布：")
for magnitude, count in magnitude_bins.items():
    print(f"{magnitude}: {count}个参数")

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

# 使用L2正则化（岭回归）
ridge = Ridge(alpha=300.0, solver='svd')  # alpha是正则化强度
ridge.fit(X_train_poly3, y_train)
y_train_pred_ridge = ridge.predict(X_train_poly3)
y_test_pred_ridge = ridge.predict(X_test_poly3)

# 使用L1正则化（Lasso回归）
lasso = Lasso(alpha=300.0)
lasso.fit(X_train_poly3, y_train)
y_train_pred_lasso = lasso.predict(X_train_poly3)
y_test_pred_lasso = lasso.predict(X_test_poly3)

# 计算并打印R²分数
print("\n各模型在训练集和测试集上的R²分数:")
print("-" * 50)
print(f"{'模型':<15} {'训练集R²':>15} {'测试集R²':>15}")
print("-" * 50)
print(f"{'L2正则化(岭)'::<15} {r2_score(y_train, y_train_pred_ridge):>15.4f} {r2_score(y_test, y_test_pred_ridge):>15.4f}")
print(f"{'L1正则化(Lasso)'::<15} {r2_score(y_train, y_train_pred_lasso):>15.4f} {r2_score(y_test, y_test_pred_lasso):>15.4f}")

# 分析参数分布
def analyze_params(name, params):
    print(f"\n{name}参数分析:")
    print(f"参数总数: {len(params)}")
    print(f"非零参数数量: {np.sum(np.abs(params) > 1e-10)}")
    print(f"最大参数值: {np.max(params):.4f}")
    print(f"最小非零参数值: {np.min(np.abs(params)[np.abs(params) > 1e-10]):.4f}")
    print(f"参数均值: {np.mean(params):.4f}")
    print(f"参数标准差: {np.std(params):.4f}")

analyze_params("L2正则化(岭)", ridge.coef_)
analyze_params("L1正则化(Lasso)", lasso.coef_)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制测试集预测结果对比
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, s=30, c='k', marker='o', label='真实值')
plt.plot(range(len(y_test)), y_test_pred_ridge, 'g-', linewidth=1, label='L2正则化')
plt.plot(range(len(y_test)), y_test_pred_lasso, 'b-', linewidth=1, label='L1正则化')
plt.title('测试集上的预测结果对比')
plt.xlabel('样本索引')
plt.ylabel('房价')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 绘制参数分布对比
plt.figure(figsize=(10, 5))

# L2正则化参数
plt.subplot(1, 2, 1)
plt.stem(range(len(ridge.coef_)), ridge.coef_, markerfmt=' ', linefmt='g-', basefmt='k-')
plt.title('L2正则化参数分布')
plt.xlabel('参数索引')
plt.ylabel('参数值')
plt.ylim(-0.1, 0.1)  # 限制y轴范围以便观察

# L1正则化参数
plt.subplot(1, 2, 2)
plt.stem(range(len(lasso.coef_)), lasso.coef_, markerfmt=' ', linefmt='b-', basefmt='k-')
plt.title('L1正则化参数分布')
plt.xlabel('参数索引')
plt.ylabel('参数值')
plt.ylim(-0.1, 0.1)

plt.tight_layout()
plt.show()