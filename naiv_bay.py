# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def naivBayes(features_selected):
    # Importing the dataset
    dataset = pd.read_csv('german_credit_data_processed.csv')

    #X = dataset.iloc[:, 1:20].values
    X = dataset.iloc[:, features_selected].values
    y = dataset.iloc[:, 21].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy * 100:.2f}%")  # 打印准确率

    precision = precision_score(y_test, y_pred)  # 用默认计算精确率
    print(f"精确率: {precision:.2f}")  # 打印精确率

    # 制作混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:\n", cm)
