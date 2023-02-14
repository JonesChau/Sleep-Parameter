'''

editor: Jones
date: 20230214
content: SVM Train 

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# support vector machine
def support_vector_machine():

    train_feature_data = pd.read_csv('../data/sitting lying posture data/sitting and lying posture feature train data.csv')

    # X = train_feature_data[['points ratio as 3']]
    X = train_feature_data[['average value', 'histogram variance', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]
    y = train_feature_data[['target']]

    # 切分 train:test = 2:1 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
    # Z = test_feature_data[['average value', 'histogram variance', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]

    # Normatlization
    sc = StandardScaler()
    scaler = sc.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # svm
    svm = SVC(C=1.0, kernel='rbf', gamma = 'auto')
    clf = svm.fit(X_train_std, y_train)
    print('weight = ',clf.dual_coef_)
    print('b = ',clf.intercept_)
    print('Indices of support vectors = ', clf.support_)
    print('Support vectors = ', clf.support_vectors_)
    print('Number of support vectors for each class = ', clf.n_support_)
    print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
    print('Support vectors = ', clf.support_vectors_.shape)
    # joblib.dump(clf, "../model/Sitting Lying Posture Classification Model/Single Feature Model/points ratio as 3 svm model.m")

    # 預測
    predict_result = svm.predict(X_test_std)
    really_result = y_test['target'].values
    print('clf:', clf.score(X_test_std, y_test))

    # 錯誤統計
    error = 0
    for i, v in enumerate(predict_result):
        if v != really_result[i]:
            error += 1
    print('test error =',error)

    count0 = 0
    count1 = 0
    sitting_index_list = []
    lying_index_list = []
    sitting_lying_list = []
    # predict_result
    for index, element in enumerate(predict_result):
        if element == 0:
            count0 += 1
            sitting_index_list.append(index)
            sitting_lying_list.append(0)
        elif element == 1:
            count1 += 1
            lying_index_list.append(index)
            sitting_lying_list.append(1)

# # 主程序
def main():
    support_vector_machine()

# 程式起點
if __name__ == '__main__':
    main()








