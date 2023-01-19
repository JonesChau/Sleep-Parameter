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


# 門檻值 22
def threshold(raw_data_array):

	threshold = 22
	raw_data_array[raw_data_array <= threshold] = 0
	return raw_data_array

# 二值化
def binarization(binarized_array):

	binarized = 0
	binarized_array[binarized_array <= binarized] = 0
	binarized_array[binarized_array > binarized] = 1023

	return binarized_array

# 非零的壓力值陣列
def nonzero_pressure_value(threshold_array):

	nonzero_array = threshold_array[np.nonzero(threshold_array > 0)]
	return nonzero_array

# 一維 array 重心位置
def center_of_mass(one_dimensional_array):

	weight_sum = 0
	for index, value in enumerate(one_dimensional_array):
		weight_sum = weight_sum + index * value

	mass_center = weight_sum/np.sum(one_dimensional_array)
	return mass_center

# 二維 array 重心位置
def two_dimension_center_of_mass(two_dimensional_array):

	# 垂直轴阵列
	vertical_axis_array = np.sum(two_dimensional_array, axis = 1)

	# 水平軸阵列
	horizontal_axis_array = np.sum(two_dimensional_array, axis = 0)

	y_axis_mc = center_of_mass(vertical_axis_array)
	x_axis_mc = center_of_mass(horizontal_axis_array)

	return x_axis_mc, y_axis_mc

def leave_bed_state(raw_data):

	leave_bed_list = pd.DataFrame(raw_data, columns= ['leave bed']).to_numpy().flatten()
	print(leave_bed_list)

	leave_bed_list_diff = np.diff(leave_bed_list)

	row = 20
	col = 11

	leave_bed = 1
	no_leave_bed = 0
	number_of_time_left_bed = 0
	leave_bed_time_list = []

	for index, element in enumerate(leave_bed_list_diff):
		if element == 1:
			# print(index)
			leave_bed_start_time = index
		if element == -1:
			# print(index)
			leave_bed_stop_time = index

			leave_bed_time = leave_bed_stop_time - leave_bed_start_time
			# print('leave_bed_time =', leave_bed_time)
			leave_bed_time_list.append(leave_bed_time)
			number_of_time_left_bed = number_of_time_left_bed + 1
			# print('number_of_time_left_bed =', number_of_time_left_bed)

	return leave_bed_time_list, number_of_time_left_bed

# 侵蝕與膨脹
def erosion_dilation(my_array):

	orignal_array = my_array[:220].reshape(20, 11)

	bit_array = (orignal_array/4).astype(np.uint8)
	binarization_array = binarization(bit_array)
	res_array1 = cv2.resize(binarization_array, (33, 60), interpolation = cv2.INTER_LINEAR)
	# print(res_array1)
	kernel_1 = np.ones((7, 7),np.uint8)
	kernel_2 = np.ones((7, 7),np.uint8)
	erosion = cv2.erode(res_array1,kernel_1,iterations = 1)
	# print(erosion)
	dilation = cv2.dilate(erosion,kernel_2,iterations = 1)
	# print(res_array1)
	# opening = cv2.morphologyEx(res_array1, cv2.MORPH_OPEN, kernel_1)
	# print(opening)
	dilation_array1 = cv2.resize(dilation, (11, 20), interpolation = cv2.INTER_LINEAR)
	# print(dilation_array1)

	new_array = np.zeros((20, 11))
	# print(new_array)

	for row, row_element in enumerate(dilation_array1):
		for col, col_element in enumerate(row_element):
			if col_element > 0:
				new_array[row][col] = orignal_array[row][col]

	# print(new_array)
	return new_array






