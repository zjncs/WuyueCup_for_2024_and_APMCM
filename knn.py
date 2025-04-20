import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from matplotlib.colors import ListedColormap

# 导入数据集
def KNN(selected_features):
    dataset = pd.read_csv('german_credit_data_processed.csv')

    # 选择特征和标签
    #X = dataset.iloc[:, 1:20].values
    X = dataset.iloc[:, selected_features].values
    y = dataset.iloc[:, 21].values

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # 特征缩放
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 拟合 K-NN 到训练集
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # 使用测试集进行预测
    y_pred = classifier.predict(X_test)

    # 计算准确率和精确率
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')

    # 打印结果
    print(f"准确率: {accuracy:.2f}")  # 打印准确率
    print(f"精确率: {precision:.2f}")  # 打印精确率

    # 制作混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:\n", cm)


