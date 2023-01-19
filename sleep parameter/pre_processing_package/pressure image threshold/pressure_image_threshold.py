'''

editor: Jones
date: 2019/12/25
content: 
方法：壓力影像權重比例
具體做法：
1.將影像轉直方圖
2.尋找直方圖中兩側邊界
3.然後從壓力值大的邊界開始往前面掃，壓力值依次疊加
4.再計算疊加壓力值與壓力值總和的佔比，即比例。
5.比例從90%到99%，再選擇合適的比例，沒超過該比例的當前壓力值即為門檻值
6.將低於門檻值的壓力值變為0，等於或大於的則保持其原來的壓力值，則得到新的壓力影像（New Pressure Image），本論文定的門檻值設定為22

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd 
import math
import pandas as pd

# 權重比例
def weighting_sum_ratio(my_array):

	hist, bin_edges = np.histogram(my_array, bins = range(0, 1025))

	i = 0
	weight_sum = 0
	while i < len(hist):
		weight_sum = hist[i] * i + weight_sum
		i += 1

	threshold_ratio = 0.99
	add_value = 0.01
	threshold_list = []

	while threshold_ratio > 0.89:
		weight_small = weight_sum * threshold_ratio
		j = 1023
		threshold = 0
		weight = 0

		while j > 0:
			weight = hist[j]*j + weight
			if weight >= weight_small:
				threshold = j
				break
			j = j - 1

		threshold_list.append(threshold)
		threshold_ratio = threshold_ratio - add_value
	return np.array(threshold_list)


# 主程序
def main():

	# open csv file 
	sitting_data_path = 'csvdata/20200113 csvData/20200113whole_sitting.csv'
	sitting_data =  pd.read_csv(sitting_data_path).to_numpy()

	face_up_data_path = 'csvdata/20200113 csvData/20200113wholeFaceUp.csv'
	face_up_data = pd.read_csv(face_up_data_path).to_numpy()

	face_right_data_path = 'csvdata/20200113 csvData/20200113wholeFaceRight.csv'
	face_right_data = pd.read_csv(face_right_data_path).to_numpy()

	face_left_data_path = 'csvdata/20200113 csvData/20200113wholeFaceLeft.csv'
	face_left_data = pd.read_csv(face_left_data_path).to_numpy()

	face_down_data_path = 'csvdata/20200113 csvData/20200113wholeFaceDown.csv'
	face_down_data = pd.read_csv(face_down_data_path).to_numpy()

	# sitting data
	sitting_threshold_array = np.zeros((1800,10))
	sitting_count = 0
	while sitting_count < len(sitting_threshold_array):
		sitting_threshold_array[sitting_count] = weighting_sum_ratio(sitting_data[sitting_count][:220])
		sitting_count = sitting_count + 1

	# face up data
	face_up_threshold_array = np.zeros((1800,10))
	face_up_count = 0
	while face_up_count < len(face_up_threshold_array):
		face_up_threshold_array[face_up_count] = weighting_sum_ratio(face_up_data[face_up_count][:220])
		face_up_count = face_up_count + 1

	# face right data
	face_right_threshold_array = np.zeros((1800,10))
	face_right_count = 0
	while face_right_count < len(face_right_threshold_array):
		face_right_threshold_array[face_right_count] = weighting_sum_ratio(face_right_data[face_right_count][:220])
		face_right_count = face_right_count + 1

	# face left data
	face_left_threshold_array = np.zeros((1800,10))
	face_left_count = 0
	while face_left_count < len(face_left_threshold_array):
		face_left_threshold_array[face_left_count] = weighting_sum_ratio(face_left_data[face_left_count][:220])
		face_left_count = face_left_count + 1

	# face down data
	face_down_threshold_array = np.zeros((1800,10))
	face_down_count = 0
	while face_down_count < len(face_down_threshold_array):
		face_down_threshold_array[face_down_count] = weighting_sum_ratio(face_down_data[face_down_count][:220])
		face_down_count = face_down_count + 1

	sitting_df = pd.DataFrame(sitting_threshold_array)
	face_up_df = pd.DataFrame(face_up_threshold_array)
	face_right_df = pd.DataFrame(face_right_threshold_array)
	face_left_df = pd.DataFrame(face_left_threshold_array)
	face_down_df = pd.DataFrame(face_down_threshold_array)

	# sitting_df.to_excel("csvData/20200114wholeThreshold csv V1.1/20200114wholeSittingThreshold.xlsx")
	# face_up_df.to_excel("csvData/20200114wholeThreshold csv V1.1/20200114wholeFaceUpThreshold.xlsx")
	# face_right_df.to_excel("csvdata/20200114wholeThreshold csv V1.1/20200114wholeFaceRightThreshold.xlsx")
	# face_left_df.to_excel("csvdata/20200114wholeThreshold csv V1.1/20200114wholeFaceLeftThreshold.xlsx")
	# face_down_df.to_excel("csvdata/20200114wholeThreshold csv V1.1/20200114wholeFaceDownThreshold.xlsx")

if __name__ == '__main__':
	main()