'''

editor: Jones
date: 20230118
content: main function

'''

import numpy as np
import pandas as pd
import os
from pre_processing_package.sitting_lying_posture_recognition import sitting_lying_feature_extraction, support_vector_machine
from pre_processing_package.affine_transformation import select_body_part, head_foot_diff, rotation_angle, vertical_axis_translate, horizontal_axis_translate, image_rotation, affine_transformation
from sleep_posture_classification_package.sleep_posture_classification import load_testing_data, image_rotated_array, normalization_data, load_model, predict_sleep_posture_result
from keras import utils as np_utils

# 坐臥姿態結果統計
def sitting_lying_result_statistics(sitting_lying_data, sitting_lying_really_label_list):

	# 提取特徵
	sitting_lying_feature_test_df = sitting_lying_feature_extraction(sitting_lying_data)
	# svm
	predict_result, sitting_lying_list, lying_index_list = support_vector_machine(sitting_lying_feature_test_df)

	really_sitting_list = []
	really_lying_list = []
	predict_accurate_sitting_count = 0
	predict_error_sitting_count = 0
	predict_accurate_lying_count = 0
	predict_error_lying_count = 0
	# 錯誤統計
	error = 0
	for index, element in enumerate(sitting_lying_really_label_list):
		if element == 0:
			really_sitting_list.append(element)
			if element == predict_result[index]:
				predict_accurate_sitting_count = predict_accurate_sitting_count + 1
			elif element != predict_result[index]:
				print('index =', index)
				predict_error_sitting_count = predict_error_sitting_count + 1

		elif element == 1:
			really_lying_list.append(element)
			if element == predict_result[index]:
				predict_accurate_lying_count = predict_accurate_lying_count + 1
			elif element != predict_result[index]:
				print('index =', index)
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


# 睡姿結果統計
def sleep_posture_result_statistics(raw_data_array):

	print("Loading testing data and label.")
	sleep_posture_test_data, sleep_posture_label_list = load_testing_data(raw_data_array)
	print('size =', len(sleep_posture_test_data))
	rotated_test_data = image_rotated_array(sleep_posture_test_data)
	print('size =', len(rotated_test_data))

	test_data_norm = normalization_data(rotated_test_data)
	label_norm = np_utils.to_categorical(sleep_posture_label_list)

	model = load_model()

	# 預測分類機率分佈結果
	predict_test = model.predict(test_data_norm)
	predict = np.argmax(predict_test,axis=1)

	# 預測分類結果
	predict_x=model.predict(test_data_norm) 
	classes_x=np.argmax(predict_x,axis=1)
	predict_result = classes_x

	really_result = sleep_posture_label_list

	error_list = []
	# 錯誤統計
	error = 0
	error_index_list = []
	predict_error_list = []
	really_error_list = []
	for i, v in enumerate(predict_result):
		if v != really_result[i]:
			predict_error_list.append(v)
			error_index_list.append(i)
			really_error_list.append(really_result[i])
			error += 1
	print('error =',error)

	predict_sleep_posture_result(really_result, predict_result)


# 主程序
def main():

	# open file 
	raw_data_df = pd.read_csv('../data/allnightData/Subject1_0430.csv')
	raw_data_array = raw_data_df.to_numpy()

	# sitting lying posture data
	# label 0: sitting posture
	# label 1: lying posture
	sitting_lying_label_list = []
	sitting_lying_data = []

	for index, element in enumerate(raw_data_array):
		if element[-2] == 0 or element[-2] == 1:
			sitting_lying_label_list.append(element[-2])
			sitting_lying_data.append(element)

	print('len =', len(sitting_lying_data))
	sitting_lying_data = np.array(sitting_lying_data)
	# 坐臥姿態結果統計
	sitting_lying_result_statistics(sitting_lying_data, sitting_lying_label_list)

	# 睡姿結果統計
	sleep_posture_result_statistics(raw_data_array)

	
# 程式起點
if __name__ == '__main__':
	main()
