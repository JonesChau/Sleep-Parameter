'''

editor: Jones
date: 20230118
content: data pre-processing

坐臥姿態區分：5個特徵值
1.計算非零壓力值的加權平均值
2.計算非零壓力值的變異數
3.找垂直轴重心，垂直軸的變異數
4.以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值*個數與壓力總和的比例，找合適半徑
5.以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值點數與總的壓力值點數的比例
6.壓力點數
7.将上述特徵值放進支撐向量機進行訓練，用於區分坐臥姿態

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import csv
import joblib
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
import cv2
from matplotlib.colors import ListedColormap
from pre_processing_package.base_function import threshold, binarization, nonzero_pressure_value, center_of_mass, two_dimension_center_of_mass
from pre_processing_package.sitting_lying_feature_extraction import nonzero_average_pressure_value, nonzero_variance_pressure_value, variance_y_axis, ratio_of_points,ratio_of_pressure_value

# 特徵值提取
def sitting_lying_feature_extraction(file_array):

	row = 20
	col = 11
	# 非零壓力值的加權平均值list
	average_pres_val_list = []
	# 直方圖變異數list
	histogram_var_list = []
	# 垂直軸變異數 list
	variance_y_axis_list = []
	# 定圓心，半徑內壓力總和/整張影像壓力總和 list
	ratio_of_pressure_value_list = []
	# 定圓心，半徑內壓力點數/整張影像壓力點數 list
	ratio_of_points_list = []

	# 提取特徵值
	for index, element in enumerate(file_array):
		item_array = np.array(element)[:220]
		item_array = item_array.reshape(row, col)
		threshold_array = threshold(item_array)
		nonzero_item_array = nonzero_pressure_value(item_array)
		average_pres_val = nonzero_average_pressure_value(nonzero_item_array)
		histogram_var = nonzero_variance_pressure_value(nonzero_item_array)
		var_y_axis = variance_y_axis(item_array)
		pres_val_ratio_list = ratio_of_pressure_value(item_array)
		ratio_list = ratio_of_points(item_array)

		average_pres_val_list.append(average_pres_val)
		histogram_var_list.append(histogram_var)
		variance_y_axis_list.append(var_y_axis)
		ratio_of_pressure_value_list.append(pres_val_ratio_list)
		ratio_of_points_list.append(ratio_list)

	# # 取半径为3
	pressure_value_index_list = []
	for index in ratio_of_pressure_value_list:
		# print('len =', len(index))
		if len(index) > 3:
			pressure_value_index_list.append(index[3])
		else:
			pressure_value_index_list.append(index[-1])

	# 取半径为3
	point_index_list = []
	for points in ratio_of_points_list:
		# print('point =', len(points))
		if len(points) > 3:
			point_index_list.append(points[3])
		else:
			point_index_list.append(points[-1])

	d = {'average value': np.array(average_pres_val_list), 'histogram variance': np.array(histogram_var_list), 
			'y axis variance': variance_y_axis_list, 'pressure value ratio as 3': np.array(pressure_value_index_list), 
			'points ratio as 3': np.array(point_index_list)}

	df = pd.DataFrame(data = d)
	print(df.head)
	return df

# 最佳超平面顯示
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0],
					y=X[y == cl, 1],
					alpha=0.6, 
					c=cmap(idx),
					edgecolor='black',
					marker=markers[idx],
					label=cl)

	# highlight test samples
	if test_idx:
		# plot all samples
		if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
			X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
			warnings.warn('Please update to NumPy 1.9.0 or newer')
		else:
			X_test, y_test = X[test_idx, :], y[test_idx]

		plt.scatter(X_test[:, 0],
					X_test[:, 1],
					c='',
					alpha=1.0,
					edgecolor='black',
					linewidths=1,
					marker='o',
					s=55, label='test set')

# support vector machine
def support_vector_machine(test_feature_data):

	train_feature_data = pd.read_csv('../data/input/sitting and lying posture feature train data.csv')

	X = train_feature_data[['average value', 'histogram variance', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]
	y = train_feature_data[['target']]

	# 切分 train:test = 2:1 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
	Z = test_feature_data[['average value', 'histogram variance', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]

	# Normatlization
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)
	Z_test_std = sc.transform(Z)

	# svm
	svm = SVC(C=1.0, kernel='rbf', gamma = 'auto')
	clf = svm.fit(X_train_std, y_train)
	# print('weight = ',clf.dual_coef_)
	# print('b = ',clf.intercept_)
	# print('Indices of support vectors = ', clf.support_)
	# print('Support vectors = ', clf.support_vectors_)
	# print('Number of support vectors for each class = ', clf.n_support_)
	# print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
	# print('Support vectors = ', clf.support_vectors_.shape)

	# 預測
	predict_result = svm.predict(X_test_std)
	really_result = y_test['target'].values
	# print('clf:', clf.score(X_test_std, y_test))

	# 錯誤統計
	error = 0
	for i, v in enumerate(predict_result):
		if v != really_result[i]:
			error += 1

	predict_result2 = svm.predict(Z_test_std)

	# print('clf:', clf.score(X_test_std, y_test))
	count0 = 0
	count1 = 0
	sitting_index_list = []
	lying_index_list = []
	sitting_lying_list = []
	# predict_result
	for index, element in enumerate(predict_result2):
		if element == 0:
			count0 += 1
			sitting_index_list.append(index)
			sitting_lying_list.append(0)
		elif element == 1:
			count1 += 1
			lying_index_list.append(index)
			sitting_lying_list.append(1)

	# print('sitting time =',count0)
	# print('lying time =', count1)
	# print('lying_index_list size =', len(lying_index_list))
	# print('sitting_lying_list =', sitting_lying_list)

	# 最佳超平面
	# plot_decision_regions(X_train_std, y_train['target'].values, classifier=svm)
	# plt.xlabel('Pressure average value [standardized]')
	# plt.ylabel('Histogram variance [standardized]')
	# plt.legend(loc='upper left')
	# plt.tight_layout()
	# plt.show()

	return predict_result2, sitting_lying_list, lying_index_list





