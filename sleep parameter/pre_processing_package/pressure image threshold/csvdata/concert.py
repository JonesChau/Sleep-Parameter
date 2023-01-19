'''

editor: Jones
date: 2019/12/12
content: 将各自的csv 组合成 一整个 train csv

'''
import pandas as pd 
import csv


# sitting data

# jones_sitting_data = pd.read_csv('1214sittingData/1204Jones_sitting.csv')
# print(len(jones_sitting_data))

# ziyang_sitting_data = pd.read_csv('1214sittingData/1210子暘_sitting.csv')
# print(len(ziyang_sitting_data))

# yucheng_sitting_data = pd.read_csv('1214sittingData/1210育誠_sitting.csv')
# print(len(yucheng_sitting_data))

# jiayun_sitting_data = pd.read_csv('1214sittingData/1210佳芸_sitting.csv')
# print(len(jiayun_sitting_data))

# jianyu_sitting_data = pd.read_csv('1214sittingData/1210建宇_sitting.csv')
# print(len(jianyu_sitting_data))

# siqi_sitting_data = pd.read_csv('1214sittingData/1210思琪_sitting.csv')
# print('len =', len(siqi_sitting_data))

# hongliang_sitting_data = pd.read_csv('1214sittingData/1210竑量_sitting.csv')
# print(len(hongliang_sitting_data))

# haosong_sitting_data = pd.read_csv('1214sittingData/1210皓菘_sitting.csv')
# print(len(haosong_sitting_data))

# pengxiang_sitting_data = pd.read_csv('1214sittingData/1210鵬翔_sitting.csv')
# print(len(pengxiang_sitting_data))

# yaotang_sitting_data = pd.read_csv('1214sittingData/1210耀堂_sitting.csv')
# print(len(yaotang_sitting_data))

# peichen_sitting_data = pd.read_csv('1214sittingData/1211沛忱_sitting.csv')
# print(len(peichen_sitting_data))

# peizhen_sitting_data = pd.read_csv('1214sittingData/1211沛臻_sitting.csv')
# print(len(peizhen_sitting_data))

# jiahong_sitting_data = pd.read_csv('1214sittingData/1211嘉宏_sitting.csv')
# print(len(jiahong_sitting_data))

# yanhong_sitting_data = pd.read_csv('1214sittingData/1211燕鴻_sitting.csv')
# print(len(yanhong_sitting_data))

# xianjun_sitting_data = pd.read_csv('1214sittingData/1211顯郡_sitting.csv')
# print(len(xianjun_sitting_data))

# zhibo_sitting_data = pd.read_csv('1214sittingData/1212郅博_sitting.csv')
# print(len(zhibo_sitting_data))

# yijia_sitting_data = pd.read_csv('1214sittingData/1212翊嘉_sitting.csv')
# print(len(yijia_sitting_data))

# bingyu_sitting_data = pd.read_csv('1214sittingData/1217秉渝_sitting.csv')
# print(len(bingyu_sitting_data))


# sitting_data = jones_sitting_data.append(ziyang_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(yucheng_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(jiayun_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(jianyu_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(siqi_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(hongliang_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(haosong_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(pengxiang_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(yaotang_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(peichen_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(peizhen_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(jiahong_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(yanhong_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(xianjun_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(zhibo_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(yijia_sitting_data, ignore_index=True)
# sitting_data = sitting_data.append(bingyu_sitting_data, ignore_index=True)
# print(sitting_data.head)

# sitting_data.to_csv("20200113whole_sitting.csv" , encoding = "utf-8")





# lying data
jones_lying_data = pd.read_csv('1216lyingData/1204Jones_lying.csv')
print(len(jones_lying_data))

ziyang_lying_data = pd.read_csv('1216lyingData/1210子暘_lying.csv')
print('len = ', len(ziyang_lying_data))

yucheng_lying_data = pd.read_csv('1216lyingData/1210育誠_lying.csv')
print('len = ', len(yucheng_lying_data))

jiayun_lying_data = pd.read_csv('1216lyingData/1210佳芸_lying.csv')
print('len = ', len(jiayun_lying_data))

jianyu_lying_data = pd.read_csv('1216lyingData/1210建宇_lying.csv')
print('len = ', len(jianyu_lying_data))

siqi_lying_data = pd.read_csv('1216lyingData/1210思琪_lying.csv')
print('len = ', len(siqi_lying_data))

hongliang_lying_data = pd.read_csv('1216lyingData/1210竑量_lying.csv')
print('len = ', len(hongliang_lying_data))

haosong_lying_data = pd.read_csv('1216lyingData/1210皓菘_lying.csv')
print('len = ', len(haosong_lying_data))

pengxiang_lying_data = pd.read_csv('1216lyingData/1210鵬翔_lying.csv')
print('len = ', len(pengxiang_lying_data))

yaotang_lying_data = pd.read_csv('1216lyingData/1210耀堂_lying.csv')
print('len = ', len(yaotang_lying_data))

peichen_lying_data = pd.read_csv('1216lyingData/1211沛忱_lying.csv')
print('len = ', len(peichen_lying_data))

peizhen_lying_data = pd.read_csv('1216lyingData/1211沛臻_lying.csv')
print('len = ', len(peizhen_lying_data))

jiahong_lying_data = pd.read_csv('1216lyingData/1211嘉宏_lying.csv')
print('len = ', len(jiahong_lying_data))

yanhong_lying_data = pd.read_csv('1216lyingData/1211燕鴻_lying.csv')
print('len = ', len(yanhong_lying_data))

xianjun_lying_data = pd.read_csv('1216lyingData/1211顯郡_lying.csv')
print('len = ', len(xianjun_lying_data))

zhibo_lying_data = pd.read_csv('1216lyingData/1212郅博_lying.csv')
print('len = ', len(zhibo_lying_data))

yijia_lying_data = pd.read_csv('1216lyingData/1212翊嘉_lying.csv')
print('len = ', len(yijia_lying_data))

bingyu_lying_data = pd.read_csv('1216lyingData/1217秉渝_lying.csv')
print('len = ', len(bingyu_lying_data))

# lying_data
lying_data = jones_lying_data.append(ziyang_lying_data, ignore_index=True)
lying_data = lying_data.append(yucheng_lying_data, ignore_index=True)
lying_data = lying_data.append(jiayun_lying_data, ignore_index=True)
lying_data = lying_data.append(jianyu_lying_data, ignore_index=True)
lying_data = lying_data.append(siqi_lying_data, ignore_index=True)
lying_data = lying_data.append(hongliang_lying_data, ignore_index=True)
lying_data = lying_data.append(haosong_lying_data, ignore_index=True)
lying_data = lying_data.append(pengxiang_lying_data, ignore_index=True)
lying_data = lying_data.append(yaotang_lying_data, ignore_index=True)
lying_data = lying_data.append(peichen_lying_data, ignore_index=True)
lying_data = lying_data.append(peizhen_lying_data, ignore_index=True)
lying_data = lying_data.append(jiahong_lying_data, ignore_index=True)
lying_data = lying_data.append(yanhong_lying_data, ignore_index=True)
lying_data = lying_data.append(xianjun_lying_data, ignore_index=True)
lying_data = lying_data.append(zhibo_lying_data, ignore_index=True)
lying_data = lying_data.append(yijia_lying_data, ignore_index=True)
lying_data = lying_data.append(bingyu_lying_data, ignore_index=True)

print(lying_data.head)
lying_data.to_csv("20200113whole_lying.csv" , encoding = "utf-8")


# sitting_data = pd.read_csv('1216whole_sitting.csv')
# lying_data = pd.read_csv('1216whole_lying.csv')

# whole_train_data = sitting_data.append(lying_data, ignore_index=True)

# print(whole_train_data)

# whole_train_data.to_csv("1216whole_train_data.csv" , encoding = "utf-8")

# bingyu_sitting_data = pd.read_csv('1214sittingData/1217秉渝_sitting.csv')
# bingyu_lying_data = pd.read_csv('1216lyingData/1217秉渝_lying.csv')
# bingyu_test_data = bingyu_sitting_data.append(bingyu_lying_data, ignore_index=True)
# print(bingyu_test_data)
# bingyu_test_data.to_csv("1217秉渝_test_data.csv" , encoding = "utf-8")



















