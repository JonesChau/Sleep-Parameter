'''

editor: Jones
date: 20230214
content: SVM Test

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# load model
clf = joblib.load("model/RBF Kernel Model/histogram variance svm model.m")

train_feature_data = pd.read_csv('input/sitting and lying posture feature train data.csv')
test_feature_data = pd.read_csv('data_csv/allnightData/allnight sitting and lying posture feature test data.csv')
# feature_input = test_feature_data[['average value', 'histogram variance', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]
X_feature_test = test_feature_data[['histogram variance']]
target = test_feature_data[['target']]

X_train = train_feature_data[['y axis variance', 'pressure value ratio as 3']]
sc = StandardScaler()
scaler = sc.fit(X_feature_test)
# X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_feature_test)
print('clf:', clf.score(X_test_std, target))

predict_result = clf.predict(X_test_std)
really_result = target.values
print('really_result =', type(really_result))
# print(type(really_result))
# # print(predict_result)
# print(len(really_result))
# 錯誤統計
really_sitting_list = []
really_lying_list = []
predict_accurate_sitting_count = 0
predict_error_sitting_count = 0
predict_accurate_lying_count = 0
predict_error_lying_count = 0
# 錯誤統計

error = 0
for index, element in enumerate(really_result):
    if element == 0:
        really_sitting_list.append(element)
        if element == predict_result[index]:
            predict_accurate_sitting_count = predict_accurate_sitting_count + 1
        elif element != predict_result[index]:
            predict_error_sitting_count = predict_error_sitting_count + 1

    elif element == 1:
        really_lying_list.append(element)
        if element == predict_result[index]:
            predict_accurate_lying_count = predict_accurate_lying_count + 1
        elif element != predict_result[index]:
            predict_error_lying_count = predict_error_lying_count + 1

print('len =', len(really_sitting_list))
print('len =', len(really_lying_list))
print('predict_accurate_sitting_count =', predict_accurate_sitting_count)
print('predict_error_sitting_count =', predict_error_sitting_count)
print('predict_accurate_lying_count =', predict_accurate_lying_count)
print('predict_error_lying_count =', predict_error_lying_count)

print('sitting accurate rate =', predict_accurate_sitting_count/len(really_sitting_list))
print('sitting error rate =', predict_error_sitting_count/len(really_sitting_list))
print('lying accurate rate =', predict_accurate_lying_count/len(really_lying_list))
print('lying error rate =', predict_error_lying_count/len(really_lying_list))








