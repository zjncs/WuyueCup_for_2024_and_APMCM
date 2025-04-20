# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # 导入支持向量机分类器
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def SVM(features_selected):
    dataset = pd.read_csv('german_credit_data_processed.csv')
    X = dataset.iloc[:, features_selected].values  # 特征变量
    #X = dataset.iloc[:, 1:20].values

    y = dataset.iloc[:, 21].values  # 标签

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # 特征缩放
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 拟合分类器到训练集
    classifier = SVC(kernel='linear')  # 创建支持向量机分类器
    classifier.fit(X_train, y_train)  # 训练模型

    # 使用测试集进行预测
    y_pred = classifier.predict(X_test)

    # 计算准确率和精确率
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # 打印结果
    print(f"准确率: {accuracy:.2f}")
    print(f"精确率: {precision:.2f}")

    # 制作混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:\n", cm)





