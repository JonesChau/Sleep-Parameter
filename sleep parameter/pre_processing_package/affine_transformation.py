'''
editor: Jones
date: 20200518
content: 
1. 影像需先做平移，後做旋轉
需先找身體的頭部、腳部，若影像中，頭部與腳部倒過來了，需將影像倒轉回來。

平移
1. 先做垂直軸的平移，後做水平軸的平移
2. 垂直軸平移，先找頭部，腳部
3. 盡量讓頭部與腳部的距離等長，即影像的身體部分盡量保留在影像的垂直軸中間
4. 水平軸平移：先找原始影像水平軸的重心，然後將影像移到水平軸的重心xc = 5

旋轉：
1. 將身體的從壓力影像中框出來，
2. 再將身體切分為三份，頭部，身體中部，腳部，
3. 再找頭部與身體中部的重心，將其做連成一直線，
4. 計算第三步連成的直線與水平軸的夾角a, if a>90, then 旋轉角度= a-90, 將影像做逆時針旋轉，旋轉後影像=原始影像*旋轉矩陣
5. if a<90, then 旋轉角度 = -(90-a), 將影像做順時針旋轉, 旋轉後影像= 原始影像*旋轉矩陣

放大：
1. 將旋轉後的影像放到至400*220
'''
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.interpolate import interp1d
import pandas as pd
import cv2
from PIL import Image
from pre_processing_package.base_function import threshold, binarization, center_of_mass, two_dimension_center_of_mass, nonzero_pressure_value

# 框出身體的部分
def select_body_part(my_array):

	# print(my_array)
	vertical_axis_array = np.sum(my_array, axis=1)
	# print(vertical_axis_array)

	# 如若人躺在床墊上，頭壓倒最高一排，腳壓倒最低一排，則first = 0, last = 20
	first = 0
	last = 20

	body_list = []
	for index, element in enumerate(vertical_axis_array):
		if element != 0:
			body_list.append(index)

	# print('body_list =', body_list)

	# 計算人壓倒床墊的那些部分，即body_list 則frist = body_list[0] last = body_list[-1] + 1
	first = body_list[0]
	last = body_list[-1]
	last = 20-last-1

	return first, last


# 頭腳區分
def head_foot_diff(my_array, first, last):

	if last == 0:
		body_array = my_array[first:]
	elif last != 0:
		body_array = my_array[first:-last]
	(middle, col) = body_array.shape

	head = 0
	foot = 0

	if(middle % 3 == 1):
		print
		head = int(middle / 3) 
		foot = int(middle / 3)
	elif(middle % 3 == 2):
		head = int(middle / 3 + 1)
		foot = int(middle / 3 + 1) 
	else:
		head = int(middle / 3) 
		foot = int(middle / 3) 

	head_array = body_array[:head]
	foot_array = body_array[-foot:]

	xg, yg = two_dimension_center_of_mass(body_array)

	y_axis_head_array = np.sum(head_array, axis = 1)
	y_axis_foot_array = np.sum(foot_array, axis = 1)
	x_axis_head_array = np.sum(head_array, axis = 0)
	x_axis_foot_array = np.sum(foot_array, axis = 0)

	y_axis_head_array_mean = np.mean(y_axis_head_array)
	y_axis_foot_array_mean = np.mean(y_axis_foot_array)
	x_axis_head_array_mean = np.mean(x_axis_head_array)
	x_axis_foot_array_mean = np.mean(x_axis_foot_array)

	y_axis_head_array_var = np.var(y_axis_head_array)
	y_axis_foot_array_var = np.var(y_axis_foot_array)
	x_axis_head_array_var = np.var(x_axis_head_array)
	x_axis_foot_array_var = np.var(x_axis_foot_array)

	if y_axis_head_array_mean > y_axis_foot_array_mean and x_axis_head_array_mean:
		return my_array
	else:
		mat180 = np.rot90(my_array, 2)
		return mat180


# pressure value added in block
def max_value_index_block_added(my_array, max_value_index):

	max_value_row_index = int(max_value_index / 11)
	max_value_col_index = max_value_index % 11
	print(max_value_row_index)
	print(max_value_col_index)
	max_value_block = my_array[max_value_row_index-1][max_value_col_index-1] + my_array[max_value_row_index-1][max_value_col_index] + my_array[max_value_row_index-1][max_value_col_index+1] + my_array[max_value_row_index][max_value_col_index-1] + my_array[max_value_row_index][max_value_col_index] + my_array[max_value_row_index][max_value_col_index+1] + my_array[max_value_row_index+1][max_value_col_index-1] + my_array[max_value_row_index+1][max_value_col_index] + my_array[max_value_row_index+1][max_value_col_index+1]

	return max_value_block


# 全身的上部，中部
def head_middle(my_array, first, last):

	if last == 0:
		body_array = my_array[first:]
	elif last != 0:
		body_array = my_array[first:-last]
	# print(body_array)
	(middle, col) = body_array.shape
	# print(body_array.shape)

	head = 0
	foot = 0

	if(middle % 3 == 1):
		head = int(middle / 3) 
		foot = int(middle / 3)
	elif(middle % 3 == 2):
		head = int(middle / 3 + 1)
		foot = int(middle / 3 + 1) 
	else:
		head = int(middle / 3) 
		foot = int(middle / 3) 

	head_array = body_array[:head]
	middle_array = body_array[head:-foot]

	# print('head =', head)

	# print('head_array =', head_array)
	# print('middle_array =', middle_array)
	head_max_value_index = np.argmax(head_array)
	middle_max_value_index = np.argmax(middle_array)

	# print('head_max_value_index =', head_max_value_index)
	# print('middle_max_value_index =', middle_max_value_index)
	middle_max_value_index = head * 11 + middle_max_value_index
	# print('middle_max_value_index =', middle_max_value_index)

	head_max_value_block = max_value_index_block_added(body_array, 60)
	middle_max_value_block = max_value_index_block_added(body_array, middle_max_value_index)

	return head_max_value_block, middle_max_value_block


# 垂直軸平移
def vertical_axis_translate(my_array):

	# print(my_array)
	xc, yc = two_dimension_center_of_mass(my_array)
	first, last = select_body_part(my_array)
	vertical_axis = np.sum(my_array, axis = 1)
	# print(vertical_axis)

	top = 0
	middle = 0
	down = -1
	count = 1
	while count < len(vertical_axis):
		if vertical_axis[count-1] != 0:
			top = count - 1
			middle = count
			break
		count = count + 1

	size = 19
	while size > 0:
		if vertical_axis[size] != 0:
			down = size
			break
		elif vertical_axis[size] == 0:
			if vertical_axis[size-1] - vertical_axis[size] != 0:
				down = size
				middle = size-1
				break
		size = size - 1

	# print('down =', down)
	# print('top =', top)
	middle = down-top
	# print('middle =', middle)

	if (middle%2) == 0:
		if top == 20-down:
			return my_array
		else:
			move = int(((20-down)-top)/2)
			# print('move =', move)
			trans_array = np.roll(my_array, move, axis = 0)
			return trans_array
	elif (middle%2) == 1:
		if (top-(20-down))==-1:
			return my_array
		elif (top-(20-down))==1:
			move = int(top-(20-down))
			# print('move =', move)
			trans_array = np.roll(my_array, move, axis = 0)
			return trans_array
		else:
			move = int(((20-down)-top)/2)
			# print('move =', move)
			trans_array = np.roll(my_array, move, axis = 0)
			return trans_array

# 水平軸平移
def horizontal_axis_translate(my_array):

	xc, yc = two_dimension_center_of_mass(my_array)
	first, last = select_body_part(my_array)

	if first == 0 and last == 0:
		cy = 0
	elif last == 0:
		body_array = my_array[first:]
		xg, yg = two_dimension_center_of_mass(body_array)
		body_part = 20 - (first + last)
		new_yc = yg * 19 / (body_part-1)
		cy = new_yc - yc
	else: 
		body_array = my_array[first:-last]
		xg, yg = two_dimension_center_of_mass(body_array)
		body_part = 20 - (first + last)
		new_yc = yg * 19 / (body_part-1)
		cy = new_yc - yc

	(h, w) = my_array.shape
	# 平移距離
	cx = 5-xc

	M = np.float32([[1,0,cx],[0,1,cy]])
	quotient_my_array = (my_array/4).astype(np.uint8)
	remainder_array = (my_array%4).astype(np.uint8)

	quotient_rotated_array = cv2.warpAffine(quotient_my_array, M, (w, h))
	remainder_rotated_array = cv2.warpAffine(remainder_array, M, (w, h))

	translate_array = quotient_rotated_array.astype(np.uint16) * 4 + remainder_rotated_array.astype(np.uint16)
	return translate_array


# 獲取旋轉角度
def rotation_angle(my_array):

	first, last = select_body_part(my_array)

	xg, yg = two_dimension_center_of_mass(my_array)

	if last == 0:
		body_array = my_array[first:]
	elif last != 0:
		body_array = my_array[first:-last]
	(middle, col) = body_array.shape

	head = 0
	foot = 0

	if(middle % 3 == 1):
		print
		head = int(middle / 3) 
		foot = int(middle / 3)
	elif(middle % 3 == 2):
		head = int(middle / 3 + 1)
		foot = int(middle / 3 + 1) 
	else:
		head = int(middle / 3) 
		foot = int(middle / 3) 

	head_array = body_array[:head]
	middle_array = body_array[head:-foot]

	head_xg, head_yg = two_dimension_center_of_mass(head_array)
	head_yg = head_yg + first
	# print('head_yg =', head_yg)
	middle_xg, middle_yg = two_dimension_center_of_mass(middle_array)
	# print('middle_yg =', middle_yg)
	middle_yg = middle_yg + head + first
	# print(middle_yg)

	a = np.array([middle_xg - head_xg, middle_yg - head_yg])
	c = np.array([[10], [0]])
	# print(np.vdot(a,c))
	# print((math.sqrt(a[0] **2 + a[1] ** 2) * math.sqrt(c[0][0] **2 + c[1][0] ** 2)))

	cos_angle = np.vdot(a,c) / (math.sqrt(a[0] **2 + a[1] ** 2) * math.sqrt(c[0][0] **2 + c[1][0] ** 2))
	# print('cos_angle =', cos_angle)

	if cos_angle > 0:
		angle = 90 - math.acos(cos_angle) * 180 /math.pi
		return -angle
		# print(-angle)
	elif cos_angle < 0:
		angle = math.acos(cos_angle) * 180 /math.pi - 90
		# print(angle)
		return angle

	# return angle, head_xg, head_yg, middle_xg, middle_yg


# 影像旋轉
def image_rotation(my_array, angle):

	(h, w) = my_array.shape
	center = (w//2, h//2)
	# print('center =', center)

	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	# print(M)
	quotient_my_array = (my_array/4).astype(np.uint8)
	remainder_array = (my_array%4).astype(np.uint8)
	# print(my_array)
	quotient_rotated_array = cv2.warpAffine(quotient_my_array, M, (w, h)) 
	remainder_rotated_array = cv2.warpAffine(remainder_array, M, (w, h))
	rotated_array = quotient_rotated_array.astype(np.uint16) * 4 + remainder_rotated_array.astype(np.uint16)
	# print(rotated_array)

	return rotated_array


# image cut 裁切人的形状
def img_cut_people_shape(my_array):

	my_array = threshold(my_array)
	# print(my_array)

	x_start = 0
	x_end = 0
	y_start = 0
	y_end = 0

	width = 0 
	h = 0

	x_axis_added_array = np.sum(my_array, axis=1)
	y_axis_added_array = np.sum(my_array, axis=0)

	x_count = 0
	while x_count < len(x_axis_added_array):
		if x_axis_added_array[x_count] != 0:
			x_start = x_count
			break
		x_count = x_count + 1
	print('x_start =', x_start)

	x_count = 0
	while x_count < len(x_axis_added_array):
		if x_axis_added_array[x_count] != 0:
			x_end = x_count
		x_count = x_count + 1
	print('x_end =', x_end)

	y_count = 0
	while y_count < len(y_axis_added_array):
		if y_axis_added_array[y_count] != 0:
			y_start = y_count
			break
		y_count = y_count + 1
	print('y_start =', y_start)

	y_count = 0
	while y_count < len(y_axis_added_array):
		if y_axis_added_array[y_count] != 0:
			y_end = y_count
		y_count = y_count + 1
	print('y_end =', y_end)

	img_cut_array = my_array[x_start:x_end+1, y_start:y_end+1]
	print(img_cut_array)

	return img_cut_array


# image resize
def img_resize(my_array):

	# my_array = thresholding(my_array)
	# print('my_array =', my_array)

	input_array = (my_array/4).astype(np.uint8)
	# print('input_array =', input_array)
	resize_array = cv2.resize(input_array, (99, 180), interpolation=cv2.INTER_LINEAR)
	# print(resize_array)
	return resize_array



# 直方圖等化 直接使用OpenCv adaptive histogram equalization
def cv2_equalizeHist(my_array):

	# my_array = my_array[my_array > 0]
	# input_array = (my_array/4).astype(np.uint8)
	# input_array = input_array[input_array > 0]
	# print(my_array)
	# print('mean =', np.mean(my_array))
	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(my_array)
	# print(len(cl1))
	# print('equ =', cl1)

	# hist, bin_edges = np.histogram(cl1, bins = range(0,255))

	# print('hist =', hist)

	# print('mena_1 =', np.mean(cl1))
	# print(np.var(cl1))
	# print(np.std(cl1))
	# print('bin_edges =',bin_edges)

	return cl1


# 仿射變換
def affine_transformation(my_array):

	# 過門檻值
	original_array = threshold(my_array.reshape(20,11))
	# print(original_array)
	# 框出壓力影像上身體的部分
	first, last = select_body_part(original_array)
	# 確定是睡床頭或者床尾
	head_foot_array = head_foot_diff(original_array, first, last)
	# print(head_foot_array)
	# 再做水平軸平移
	horzial_array = horizontal_axis_translate(head_foot_array)
	# print(horzial_array)
	# 知悉旋轉角度
	angle = rotation_angle(horzial_array)
	# print('angle =', angle)
	# 做旋轉
	rotated_array = image_rotation(horzial_array, angle)
	# print(rotated_array)
	return rotated_array


# 同一個睡姿睡姿壓力值疊加
def same_sleep_posture_added(raw_data):

	# same sleep posture pressure value add
	same_sleep_posture_array = np.zeros((20, 11))

	count = 0
	for col_index, col_element in enumerate(raw_data):
		if col_element[-1] == 0:
			rotated_array = affine_transformation(col_element[:220])
			same_sleep_posture_array = same_sleep_posture_array + rotated_array
			# face_up_list.append(rotated_array)
			count = count + 1
	print('count =', count)

	return same_sleep_posture_array, count

def variance_same_sleep_posture(average_same_sleep_posture_array):

	average_same_sleep_posture_array_thresold = threshold(average_same_sleep_posture_array)
	nonzero_array = nonzero_pressure_value(average_same_sleep_posture_array_thresold)
	average_pressure_value = nonzero_average_pressure_value(nonzero_array)
	same_sleep_posture_variance = nonzero_variance_pressure_value(nonzero_array, average_pressure_value)
	return same_sleep_posture_variance


class MyGaussianBlur():
    #初始化
    def __init__(self, radius=1, sigema=1.5):
        self.radius=radius
        self.sigema=sigema    
    #高斯的計算公式
    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
    #得到濾波模版
    def template(self):
        sideLength=self.radius*2+1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i,j]=self.calc(i-self.radius, j-self.radius)
        all=result.sum()
        return result/all    
    #濾波函式
    def filter(self, image, template, radius): 
        arr=np.array(image)
        height=arr.shape[0]
        width=arr.shape[1]
        newData=np.zeros((height-radius, width-radius))
        for i in range(self.radius, height-self.radius):
            for j in range(self.radius, width-self.radius):
                t=arr[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1]
                a= np.multiply(t, template)
                newData[i, j] = a.sum()

        new_img = np.zeros((height-(radius*2), width-(radius*2)))

        for col_index, col_element in enumerate(new_img):
        	for row_index, row_element in enumerate(col_element):
        		new_img[col_index][row_index] = newData[col_index + radius][row_index + radius]

        return new_img