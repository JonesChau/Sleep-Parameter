'''

editor: Jones
date: 20200807
content: sleep posture classification

'''

from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf 
from keras import regularizers
from pre_processing_package.affine_transformation import select_body_part, head_foot_diff, rotation_angle, vertical_axis_translate, horizontal_axis_translate, image_rotation, affine_transformation


# Load training data from a file and return X, y
def load_testing_data(raw_data_array):

	row = 20
	col = 11

	# sleep posture posture data
	# label 0: face up
	# label 1: face right
	# label 2: face left
	# label 3: face down
	sleep_posture_label_list = []
	sleep_posture_data_list = []

	for index, element in enumerate(raw_data_array):
		if element[-1] == 0 or element[-1] == 1 or element[-1] == 2 or element[-1] == 3 :
			sleep_posture_label_list.append(element[-1])
			element_reshape = element[:220].reshape(row, col)
			sleep_posture_data_list.append(element_reshape)

	sleep_posture_data = np.array(sleep_posture_data_list)
	label_array = np.array(sleep_posture_label_list)

	return sleep_posture_data, label_array

def image_rotated_array(raw_data):

	rotated_data_array = []
	count = 0
	for i in raw_data:
		i_rotated = affine_transformation(i)
		rotated_data_array.append(i_rotated)
		count = count + 1

	rotated_data_array = np.array(rotated_data_array)

	return rotated_data_array

def normalization_data(data):

	row = 20
	col = 11
	# 多加一個顏色的維度 
	data_norm = data.reshape(data.shape[0], row, col, 1).astype('float64')
	# print('data.shape[0] = ', data.shape[0])
	# 正規化
	data_norm = data_norm/1023

	return data_norm


# 預測分類結果
def predict_sleep_posture_result(really_result, predict_result):

	really_face_up_list = []
	really_face_right_list = []
	really_face_left_list = []
	really_face_down_list = []

	# really face up 
	predict_accurate_face_up_count = 0
	really_face_up_predict_face_right_count = 0
	really_face_up_predict_face_left_count = 0
	really_face_up_predict_face_down_count = 0

	# face right
	predict_accurate_face_right_count = 0
	really_face_right_predict_face_up_count = 0
	really_face_right_predict_face_left_count = 0
	really_face_right_predict_face_down_count = 0

	# face left
	predict_accurate_face_left_count = 0
	really_face_left_predict_face_up_count = 0
	really_face_left_predict_face_right_count = 0
	really_face_left_predict_face_down_count = 0

	# face down
	predict_accurate_face_down_count = 0
	really_face_down_predict_face_up_count = 0
	really_face_down_predict_face_right_count = 0
	really_face_down_predict_face_left_count = 0

	# 錯誤統計
	error = 0
	for index, element in enumerate(really_result):
		if element == 0:
			really_face_up_list.append(element)
			if element == predict_result[index]:
				predict_accurate_face_up_count = predict_accurate_face_up_count + 1
			else:
				if predict_result[index] == 1:
					really_face_up_predict_face_right_count = really_face_up_predict_face_right_count + 1
				elif predict_result[index] == 2:
					really_face_up_predict_face_left_count = really_face_up_predict_face_left_count + 1
				elif predict_result[index] == 3:
					really_face_up_predict_face_down_count = really_face_up_predict_face_down_count + 1

		elif element == 1:
			really_face_right_list.append(element)
			if element == predict_result[index]:
				predict_accurate_face_right_count = predict_accurate_face_right_count + 1
			else:
				if predict_result[index] == 0:
					really_face_right_predict_face_up_count = really_face_right_predict_face_up_count + 1
				elif predict_result[index] == 2:
					really_face_right_predict_face_left_count = really_face_right_predict_face_left_count + 1
				elif predict_result[index] == 3:
					really_face_right_predict_face_down_count = really_face_right_predict_face_down_count + 1

		elif element == 2:
			really_face_left_list.append(element)
			if element == predict_result[index]:
				predict_accurate_face_left_count = predict_accurate_face_left_count + 1
			else:
				if predict_result[index] == 0:
					really_face_left_predict_face_up_count = really_face_left_predict_face_up_count + 1
				elif predict_result[index] == 1:
					print('index =', index)
					really_face_left_predict_face_right_count = really_face_left_predict_face_right_count + 1
				elif predict_result[index] == 3:
					really_face_left_predict_face_down_count = really_face_left_predict_face_down_count + 1

		elif element == 3:
			really_face_down_list.append(element)
			if element == predict_result[index]:
				predict_accurate_face_down_count = predict_accurate_face_down_count + 1
			else:
				if predict_result[index] == 0:
					really_face_down_predict_face_up_count = really_face_down_predict_face_up_count + 1
				elif predict_result[index] == 1:
					really_face_down_predict_face_right_count = really_face_down_predict_face_right_count + 1
				elif predict_result[index] == 2:
					really_face_down_predict_face_left_count = really_face_down_predict_face_left_count + 1

	print('len =', len(really_face_up_list))
	print('predict_accurate_face_up_count =', predict_accurate_face_up_count)
	print('really_face_up_predict_face_right_count =', really_face_up_predict_face_right_count)
	print('really_face_up_predict_face_left_count =', really_face_up_predict_face_left_count)
	print('really_face_up_predict_face_down_count =', really_face_up_predict_face_down_count)

	print('len =', len(really_face_right_list))
	print('predict_accurate_face_right_count =', predict_accurate_face_right_count)
	print('really_face_right_predict_face_up_count =', really_face_right_predict_face_up_count)
	print('really_face_right_predict_face_left_count =', really_face_right_predict_face_left_count)
	print('really_face_up_predict_face_down_count =', really_face_right_predict_face_down_count)

	print('len =', len(really_face_left_list))
	print('predict_accurate_face_left_count =', predict_accurate_face_left_count)
	print('really_face_left_predict_face_up_count =', really_face_left_predict_face_up_count)
	print('really_face_left_predict_face_right_count =', really_face_left_predict_face_right_count)
	print('really_face_left_predict_face_down_count =', really_face_left_predict_face_down_count)

	print('len =', len(really_face_down_list))
	print('predict_accurate_face_down_count =', predict_accurate_face_down_count)
	print('really_face_down_predict_face_up_count =', really_face_down_predict_face_up_count)
	print('really_face_down_predict_face_right_count =', really_face_down_predict_face_right_count)
	print('really_face_down_predict_face_left_count =', really_face_down_predict_face_left_count)


# load model
def load_model():

	# 從 HDF5 檔案中載入模型
	model = tf.keras.models.load_model('../model/sleep_posture_classification_model.h5')
	# 顯示模型結構
	model.summary()
	return model


# # 從 HDF5 檔案中載入模型
# model = tf.keras.models.load_model('../model/20200828_1234.h5')
# # 顯示模型結構
# model.summary()
# 檔案
# test_file = 'csvData/allnightData/Jane0418.csv'

# print("Loading testing data and label.")
# test_data, label = load_testing_data(test_file)
# print('size =', len(test_data))
# rotated_test_data = image_rotated_array(test_data)
# print('size =', len(rotated_test_data))

# print('test_data[0] =', test_data[0:10])
# test_data_norm = normalization_data(rotated_test_data)
# print(test_data_norm[0])
# label_norm = np_utils.to_categorical(label)
# print(label_norm)


# 預測分類機率分佈結果
# predict_test = model.predict(test_data_norm)
# predict = np.argmax(predict_test,axis=1)
# print(predict_test[245:270])

# 預測分類結果
# predict_x=model.predict(test_data_norm) 
# classes_x=np.argmax(predict_x,axis=1)
# predict_result = classes_x

# predict_result = model.predict_classes(test_data_norm).astype('int')
# print(predict_result[:10]) 

# really_result = label

# print('predict_result =', predict_result)
# print('really_result =', really_result.ravel())

# error_list = []
# # 錯誤統計
# error = 0
# error_index_list = []
# predict_error_list = []
# really_error_list = []
# for i, v in enumerate(predict_result):
# 	if v != really_result[i]:
# 		# print('predict_result =', v)
# 		predict_error_list.append(v)
# 		# print('count =', i)
# 		error_index_list.append(i)
# 		# print(predict_test[i])
# 		# error_list.append(test_data[i])
# 		# print('really_result = ', really_result[i])
# 		really_error_list.append(really_result[i])
# 		error += 1
# print('error =',error)
# print(error_index_list)

# for i in really_error_list:
# 	print(i)

# predict_sleep_posture_result(really_result, predict_result)

# really_result_show_list = []

# for i in really_result:
# 	if i == 0:
# 		really_result_show_list.append('Face Up')
# 	elif i == 1:
# 		really_result_show_list.append('Face Right')
# 	elif i ==2:
# 		really_result_show_list.append('Face Left')
# 	elif i == 3:
# 		really_result_show_list.append('Face Down')


# predict_result_show_list = []

# for i in predict_result:
# 	if i == 0:
# 		predict_result_show_list.append('Face Up')
# 	elif i == 1:
# 		predict_result_show_list.append('Face Right')
# 	elif i == 2:
# 		predict_result_show_list.append('Face Left')
# 	elif i == 3:
# 		predict_result_show_list.append('Face Down')

# plt.subplot(211)
# plt.subplot()

# plt.plot(range(len(really_result_show_list)), really_result_show_list)
# plt.title("Subject 2 Really Sleep Posture in First Night", fontsize = 24)
# # 設定刻度字型大小
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# # 設定座標標籤字型大小
# plt.xlabel('Time(s)', fontsize=15)

# plt.subplot(212)
# plt.plot(range(len(predict_result_show_list)), predict_result_show_list)
# plt.title("Subject 2 Predict Sleep Posture in First Night",fontsize = 25)
# # # 設定刻度字型大小
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# # # 設定座標標籤字型大小
# plt.xlabel('Time(s)', fontsize=15)
# plt.savefig('image/predict result/Jones0421.png')
# plt.show()
