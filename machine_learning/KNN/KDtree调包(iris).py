import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = './datas/iris.data'
names = ['x1', 'x2', 'x3', 'x4', 'y']
df = pd.read_csv(path, header=None, names=names, sep=",")
print(df.head())
print(df.shape)
print(df["y"].value_counts())


# 2. 数据清洗
# NOTE: 不需要做数据处理
def parse_record(row):
    result = []
    r = zip(names, row)
    for name, value in r:
        if name == 'y':
            if value == 'Iris-setosa':
                result.append(1)
            elif value == 'Iris-versicolor':
                result.append(2)
            elif value == 'Iris-virginica':
                result.append(3)
            else:
                result.append(0)
        else:
            result.append(value)
    return result


df = df.apply(lambda row: pd.Series(parse_record(row), index=names), axis=1)
df['y'] = df['y'].astype(np.int32)
df.info()
print(df["y"].value_counts())
flag = False

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
# X = df.iloc[:,:-1]
X = df[names[0:-1]]
print(X.shape)
Y = df[names[-1]]
print(Y.shape)
print(Y.value_counts())
# sys.exit()

# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程

# 6. 模型对象的构建
"""
KNN:
    n_neighbors=5,
    weights='uniform',
    algorithm='auto', 
    leaf_size=30,
    p=2,
    metric='minkowski', 
    metric_params=None, 
    n_jobs=1
"""
KNN = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='kd_tree')

# 7. 模型的训练
KNN.fit(x_train, y_train)

# 8. 模型效果评估z
train_predict = KNN.predict(x_train)
test_predict = KNN.predict(x_test)
print("KNN算法：测试集上的效果(准确率):{}".format(KNN.score(x_test, y_test)))
print("KNN算法：训练集上的效果(准确率):{}".format(KNN.score(x_train, y_train)))
print(accuracy_score(y_true=y_train, y_pred=train_predict))

# 模型的保存与加载
import joblib

joblib.dump(KNN, "./knn.m")  # 保存模型