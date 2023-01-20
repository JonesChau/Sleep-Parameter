'''
editor: Jones
date:2020/05/07
content:
Let’s create a fuzzy control system which models how you might choose to tip at a restaurant. 
When tipping, you consider the service and food quality, rated between 0 and 10. You use this to leave a tip of between 0 and 25%.
We would formulate this problem as:

1. Antecednets (Inputs)
	1.1 service
		Universe (ie, crisp value range): How good was the service of the wait staff, on a scale of 0 to 10?
		Fuzzy set (ie, fuzzy value range): poor, acceptable, amazing
	1.2 food quality
		Universe: How tasty was the food, on a scale of 0 to 10?
		Fuzzy set: bad, decent, great
2. Consequents (Outputs)
	tip
		Universe: How much should we tip, on a scale of 0% to 25%
	Fuzzy set: low, medium, high
3. Rules
	IF the service was good or the food quality was good, THEN the tip will be high.
	IF the service was average, THEN the tip will be medium.
	IF the service was poor and the food quality was poor THEN the tip will be low.
4. Usage
	If I tell this controller that I rated:
		the service as 9.8, and
		the quality as 6.5,
	it would recommend I leave:
		a 20.2% tip.

'''


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
# Plot the result in pretty 3D with alpha blending
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


leave_bed_count = 6
body_movement_count = 100
body_movement_time_interval_count = 61
sleep_quality_count = 101
time_in_bed_time = 13

# Universe variables
# 睡眠品質參數
# 臥床時間
# 離床次數
# 整晚翻身次數
# 翻身時間間隔
# 壓力分佈


time_in_bed = ctrl.Antecedent(np.arange(0, time_in_bed_time, 1), 'Time in Bed(h)')
leave_bed = ctrl.Antecedent(np.arange(-1, leave_bed_count, 1), 'Number of Times of Bed-Leaving(times)')
body_movement = ctrl.Antecedent(np.arange(0, body_movement_count, 1), 'Number of Times of Body Movements all night(times)')
body_movement_time_interval = ctrl.Antecedent(np.arange(0, body_movement_time_interval_count, 1), 'Standard deviation of Body Movements Time Interval')
sleep_quality = ctrl.Consequent(np.arange(0, sleep_quality_count ,1), 'Sleep Quality')


# Auto-membership function population (3,5,7)
time_in_bed_names  = ['L','M','H']
leave_bed_names = ['L','H']
body_movement_names = ['L','M','H']
body_movement_time_interval_names = ['L','H']


sleep_quality_names = ['Very Bad', 'Bad', 'Fair', 'Good', 'Very Good']

# automf
time_in_bed.automf(names = time_in_bed_names)
leave_bed.automf(names = leave_bed_names)
body_movement.automf(names = body_movement_names)
body_movement_time_interval.automf(names = body_movement_time_interval_names)
# body_movement_time_interval.automf(5)
sleep_quality.automf(names = sleep_quality_names)

time_in_bed['L'] = fuzz.trapmf(time_in_bed.universe, [0, 0, 5, 7])
time_in_bed['M'] = fuzz.trapmf(time_in_bed.universe, [5, 7, 9, 11])
time_in_bed['H'] = fuzz.trapmf(time_in_bed.universe, [9, 11, 13, 13])


leave_bed['L'] = fuzz.trapmf(leave_bed.universe, [-1, -1, 1, 4])
leave_bed['H'] = fuzz.trapmf(leave_bed.universe, [1 , 4, leave_bed_count, leave_bed_count])

body_movement['L'] = fuzz.trapmf(body_movement.universe, [0, 0, 10, 35])
body_movement['M'] = fuzz.trimf(body_movement.universe, [10, 35, 60])
body_movement['H'] = fuzz.trapmf(body_movement.universe, [35 , 60, body_movement_count, body_movement_count])

body_movement_time_interval['L'] = fuzz.trapmf(body_movement_time_interval.universe, [0, 0, 10, 40])
body_movement_time_interval['H'] = fuzz.trapmf(body_movement_time_interval.universe, [10, 40, 60, 60])


# Custom triangle membership functions
sleep_quality['Very Bad'] = fuzz.trapmf(sleep_quality.universe, [0 ,0, 10, 30])
sleep_quality['Bad'] = fuzz.trimf(sleep_quality.universe, [10, 30, 50])
sleep_quality['Fair'] = fuzz.trimf(sleep_quality.universe, [30 ,50, 70])
sleep_quality['Good'] = fuzz.trimf(sleep_quality.universe, [50 ,70, 90])
sleep_quality['Very Good'] = fuzz.trapmf(sleep_quality.universe, [70 ,90, 100, 100])



#view memberships
# time_in_bed.view()
# leave_bed.view()
# body_movement.view()
# body_movement_time_interval.view()
# sleep_quality.view()
# plt.xticks(np.arange(0, 100, step=5))
# plt.show()

#Fuzzy rules
rule1 = ctrl.Rule(time_in_bed['M'] & leave_bed['L'] & body_movement['M'] & body_movement_time_interval['L'], sleep_quality['Very Good'])
rule2 = ctrl.Rule(time_in_bed['L'] & leave_bed['L'] & body_movement['M'] & body_movement_time_interval['L'], sleep_quality['Good'])
rule3 = ctrl.Rule(time_in_bed['M'] & leave_bed['H'] & body_movement['M'] & body_movement_time_interval['L'], sleep_quality['Good'])
rule4 = ctrl.Rule(time_in_bed['M'] & leave_bed['L'] & body_movement['L'] & body_movement_time_interval['H'], sleep_quality['Fair'])

rule5 = ctrl.Rule(time_in_bed['H'] & leave_bed['L'] & body_movement['M'] & body_movement_time_interval['L'], sleep_quality['Fair'])
rule6 = ctrl.Rule(time_in_bed['M'] & leave_bed['L'] & body_movement['M'] & body_movement_time_interval['H'], sleep_quality['Fair'])
rule7 = ctrl.Rule(time_in_bed['L'] & leave_bed['L'] & body_movement['M'] & body_movement_time_interval['H'], sleep_quality['Bad'])
rule8 = ctrl.Rule(time_in_bed['L'] & leave_bed['H'] & body_movement['M'] & body_movement_time_interval['L'], sleep_quality['Bad'])

rule9 = ctrl.Rule(time_in_bed['M'] & leave_bed['L'] & body_movement['L'] & body_movement_time_interval['H'], sleep_quality['Bad'])
rule10 = ctrl.Rule(time_in_bed['M'] & leave_bed['H'] & body_movement['L'] & body_movement_time_interval['L'], sleep_quality['Bad'])
rule11 = ctrl.Rule(time_in_bed['M'] & leave_bed['H'] & body_movement['M'] & body_movement_time_interval['H'], sleep_quality['Bad'])
rule12 = ctrl.Rule(time_in_bed['H'] & leave_bed['H'] & body_movement['M'] & body_movement_time_interval['L'], sleep_quality['Bad'])

rule13 = ctrl.Rule(time_in_bed['L'] & leave_bed['L'] & body_movement['L'], sleep_quality['Very Bad'])
rule14 = ctrl.Rule(time_in_bed['L'] & leave_bed['L'] & body_movement['H'], sleep_quality['Very Bad'])
rule15 = ctrl.Rule(time_in_bed['L'] & leave_bed['H'] & body_movement['L'], sleep_quality['Very Bad'])
rule16 = ctrl.Rule(time_in_bed['L'] & leave_bed['H'] & body_movement['M'] & body_movement_time_interval['H'], sleep_quality['Very Bad'])

rule17 = ctrl.Rule(time_in_bed['L'] & leave_bed['H'] & body_movement['L'], sleep_quality['Very Bad'])
rule18 = ctrl.Rule(time_in_bed['M'] & leave_bed['L'] & body_movement['H'], sleep_quality['Very Bad'])
rule19 = ctrl.Rule(time_in_bed['M'] & leave_bed['H'] & body_movement['L'] & body_movement_time_interval['H'], sleep_quality['Very Bad'])
rule20 = ctrl.Rule(time_in_bed['M'] & leave_bed['H'] & body_movement['H'], sleep_quality['Very Bad'])

rule21 = ctrl.Rule(time_in_bed['H'] & leave_bed['L'] & body_movement['L'], sleep_quality['Very Bad'])
rule22 = ctrl.Rule(time_in_bed['H'] & body_movement['M'] & body_movement_time_interval['H'], sleep_quality['Very Bad'])
rule23 = ctrl.Rule(time_in_bed['H'] & body_movement['H'], sleep_quality['Very Bad'])
rule24 = ctrl.Rule(time_in_bed['H'] & leave_bed['H'] & body_movement['H'], sleep_quality['Very Bad'])

# rule1.view()
# rule2.view()
# rule3.view()

# #Control System Creation and Simulation
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, 
	rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule21, rule22, rule23, rule24])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)


# # We can simulate at higher resolution with full accuracy
# upsampled = np.linspace(-2, 2, 21)
# x, y = np.meshgrid(upsampled, upsampled)
# z = np.zeros_like(x)

# # Loop through the system 21*21 times to collect the control surface
# for i in range(21):
#     for j in range(21):
#         tipping.input['Time in Bed'] = x[i, j]
#         tipping.input['Number of Times of Bed-Leaving'] = y[i, j]
#         tipping.compute()
#         z[i, j] = tipping.output['output']

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
#                        linewidth=0.4, antialiased=True)

# cset = ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
# cset = ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
# cset = ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)

# ax.view_init(30, 200)

# # # # Pass inputs to the ControlSystem & compute
# Jones0420.csv
# tipping.input['Time in Bed(h)'] = 7.45
# tipping.input['Number of Times of Bed-Leaving(times)'] = 3
# tipping.input['Number of Times of Body Movements all night(times)'] = 23
# tipping.input['Standard deviation of Body Movements Time Interval'] = 15.79

# Jones0421.csv
# tipping.input['Time in Bed(h)'] = 6.89
# tipping.input['Number of Times of Bed-Leaving(times)'] = 0
# tipping.input['Number of Times of Body Movements all night(times)'] = 24
# tipping.input['Standard deviation of Body Movements Time Interval'] = 13.21

# # Jones0430.csv
tipping.input['Time in Bed(h)'] = 7.20
tipping.input['Number of Times of Bed-Leaving(times)'] = 1
tipping.input['Number of Times of Body Movements all night(times)'] = 26
tipping.input['Standard deviation of Body Movements Time Interval'] = 12.91

# # Jane0418.csv
# tipping.input['Time in Bed(h)'] = 6.88
# tipping.input['Number of Times of Bed-Leaving(times)'] = 3
# tipping.input['Number of Times of Body Movements all night(times)'] = 8
# tipping.input['Standard deviation of Body Movements Time Interval'] = 41.33

# # Jane0422.csv
# tipping.input['Time in Bed(h)'] = 7.92
# tipping.input['Number of Times of Bed-Leaving(times)'] = 2
# tipping.input['Number of Times of Body Movements all night(times)'] = 11
# tipping.input['Standard deviation of Body Movements Time Interval'] = 36.59

# # Jane0429.csv
# tipping.input['Time in Bed(h)'] = 6.34
# tipping.input['Number of Times of Bed-Leaving(times)'] = 3
# tipping.input['Number of Times of Body Movements all night(times)'] = 10
# tipping.input['Standard deviation of Body Movements Time Interval'] = 28.92


# body_movement[]
tipping.compute()

# # # #visualize & view
print(tipping.output['Sleep Quality'])
# plt.figure()
# sleep_quality.view(sim=tipping)
# plt.savefig('Jane0429_sleep_uality.png')
# plt.show()