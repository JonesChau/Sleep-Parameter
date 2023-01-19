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
from pre_processing_package.base_function import threshold, binarization, center_of_mass, two_dimension_center_of_mass, nonzero_pressure_value

# 計算非零壓力值的加權平均值
def nonzero_average_pressure_value(nonzero_array):

	average_pres_val = np.mean(nonzero_array)
	return average_pres_val

# 計算非零壓力值的變異數
def nonzero_variance_pressure_value(nonzero_array):

	histogram_variance = np.var(nonzero_array)
	return histogram_variance

# 垂直軸的變異數
def variance_y_axis(my_array):

	# 水平軸總和
	y_axis_array = np.sum(my_array, axis = 1)
	yg = center_of_mass(y_axis_array)

	var_y_axis = 0
	for y, y_axis_val in enumerate(y_axis_array):
		var_y_axis = var_y_axis + (y_axis_val / np.sum(y_axis_array) * (y - yg) ** 2)

	return var_y_axis

# 以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值點數與總的壓力值點數的比例
def ratio_of_points(my_array):

	xg, yg =  two_dimension_center_of_mass(my_array)
	binary_array = binarization(my_array)

	distance_list = []
	row = 20
	col = 11
	radius_1 = 1.0

	for row_index, row_element in enumerate(binary_array):
		for col_index, col_element in enumerate(row_element):
			if(col_element == 1023):
				distance = np.sqrt((xg - col_index) ** 2 + (yg - row_index) ** 2)
				distance_list.append(distance)

	distance_array = np.sort(np.array(distance_list))
	ratio_list = []
	count = 0

	while count <= math.ceil(max(distance_list)):
		ratio = np.size(distance_array[distance_array < radius_1 * count])/np.size(distance_array)
		ratio_list.append(round(ratio, 3))
		count += 1

	return ratio_list

# 以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值*個數與壓力總和的比例，找合適半徑
def ratio_of_pressure_value(my_array):

	xg, yg =  two_dimension_center_of_mass(my_array)

	distance_list = []
	pressure_value_list = []
	row = 20
	col = 11
	radius_1 = 1.0
	count = 0

	for row_index, row_element in enumerate(my_array):
		for col_index, col_element in enumerate(row_element):
			if(col_element > 0):
				distance = np.sqrt((xg - col_index) ** 2 + (yg - row_index) ** 2)
				distance_list.append(distance)
				pressure_value_list.append(col_element)

	pres_val_ratio_list = []
	pres_val_ratio = 0

	while count <= math.ceil(max(distance_list)):
		i = 0
		while i < len(distance_list):
			if (distance_list[i] > radius_1 * (count-1)) and (distance_list[i] < radius_1 * count):
				pres_val_ratio = pressure_value_list[i] + pres_val_ratio
			i = i+1
		value_ratio = pres_val_ratio / np.sum(my_array)
		pres_val_ratio_list.append(round(value_ratio, 3))
		count = count + 1

	return pres_val_ratio_list















