import math
import numpy as np
from keras import models


def NeurophysicalModel(velocity_1, velocity_2, velocity_3, type_surface, time, current_angle):

    current_angle = current_angle
    time = time
    input_data = np.array([velocity_1 * 30 / math.pi, velocity_2 * 30 / math.pi, velocity_3 * 30 / math.pi,
                           type_surface])

    input_data = input_data.reshape(1, -1)

    "normalization"
    mean = np.array([-0.04328374, -0.04481319,  0.08801374, 1.99927798])
    std = np.array([270.13250383, 270.11489776, 270.03726287, 0.81634887])
    input_data -= mean
    input_data /= std

    mean_velocity_current_green = np.array([-3.70464561, 1.81659984, 2.1912304])
    std_velocity_current_green = np.array([204.12715797, 194.17465242, 200.22339528])

    mean_velocity_current_grey = np.array([-3.15935342,  1.9343096, 1.65945736])
    std_velocity_current_grey = np.array([196.32761694, 199.25209513, 191.52317769])

    mean_for_slippage_green = np.array([0.73784104, 0.78476592, 0.73912061])
    std_for_slippage_green = np.array([0.31336431, 0.30759276, 0.32479139])

    mean_for_slippage_grey = np.array([0.84230211, 0.76257781, 0.85836667])
    std_for_slippage_grey = np.array([0.39570568, 0.33581775, 0.40823333])

    "predicting real speeds"

    motor_velocity_first = models.load_model('velocity_first')
    motor_velocity_second = models.load_model('velocity_second')
    motor_velocity_third = models.load_model('velocity_third')

    "prediction of the currents power consumption"

    current_first_motor_green = models.load_model('current_first_green')
    current_second_motor_green = models.load_model('current_second_green')
    current_third_motor_green = models.load_model('current_third_green')

    current_first_motor_grey = models.load_model('current_first_grey')
    current_second_motor_grey = models.load_model('current_second_grey')
    current_third_motor_grey = models.load_model('current_third_grey')

    current_first_motor_table = models.load_model('current_first_table')
    current_second_motor_table = models.load_model('current_second_table')
    current_third_motor_table = models.load_model('current_third_table')

    "prediction of the slippage coefficient"

    wheel_first_slippage_green = models.load_model('slippage_wheel_1_green')
    wheel_second_slippage_green = models.load_model('slippage_wheel_2_green')
    wheel_third_slippage_green = models.load_model('slippage_wheel_3_green')

    wheel_first_slippage_grey = models.load_model('slippage_wheel_1')
    wheel_second_slippage_grey = models.load_model('slippage_wheel_2')
    wheel_third_slippage_grey = models.load_model('slippage_wheel_3')

    "STRUCTURE"
    speed_motor_1 = motor_velocity_first.predict(input_data)
    speed_motor_2 = motor_velocity_second.predict(input_data)
    speed_motor_3 = motor_velocity_third.predict(input_data)

    motors_speed = np.c_[speed_motor_1, speed_motor_2]
    motors_speed = np.c_[motors_speed, speed_motor_3]

    grey_location_for_current = np.array(motors_speed[0])

    grey_location_for_current -= mean_velocity_current_grey
    grey_location_for_current /= std_velocity_current_grey

    grey_location_for_current = grey_location_for_current.reshape(1, -1)

    current_first_motor_on_grey = current_first_motor_grey.predict(grey_location_for_current)
    current_second_motor_on_grey = current_second_motor_grey.predict(grey_location_for_current)
    current_third_motor_on_grey = current_third_motor_grey.predict(grey_location_for_current)

    current_motors_on_grey = np.c_[current_first_motor_on_grey, current_second_motor_on_grey]
    current_motors_on_grey = np.c_[current_motors_on_grey, current_third_motor_on_grey]

    current_motors_on_grey -= mean_for_slippage_grey
    current_motors_on_grey /= std_for_slippage_grey

    slippage_first_grey = wheel_first_slippage_grey.predict(current_motors_on_grey)
    slippage_second_grey = wheel_second_slippage_grey.predict(current_motors_on_grey)
    slippage_third_grey = wheel_third_slippage_grey.predict(current_motors_on_grey)

    if speed_motor_1[0] < 0:
        slippage_first_grey[0] = -slippage_first_grey[0]

    if speed_motor_2[0] < 0:
        slippage_second_grey[0] = -slippage_second_grey[0]

    if speed_motor_3[0] < 0:
        slippage_third_grey[0] = -slippage_third_grey[0]

    slippage = np.c_[slippage_first_grey, slippage_second_grey]
    slippage = np.c_[slippage, slippage_third_grey]

    real_velocity = motors_speed - slippage
    real_velocity = real_velocity * math.pi / 30

    "Radius of omni-wheel"

    r = 0.04

    "Direct kinematics"
    matrix = np.array([[-2 * math.sin(math.pi / 3), -2 * math.sin(math.pi), 2 * math.sin(math.pi / 3)],
                       [2 * math.cos(math.pi / 3), 2 * math.cos(math.pi), 2 * math.cos(math.pi / 3)],
                       [1 / 0.13, 1 / 0.13, 1 / 0.13]])

    m1 = np.dot(1/3, matrix)

    mat = np.matmul(m1, real_velocity[0]/16)

    vxvywz = np.dot(r, mat)

    dxdyda_local = np.array([vxvywz[0] * time, vxvywz[1] * time, vxvywz[2] * time])

    angle = dxdyda_local[2]

    if -3 <= velocity_1 <= 3 and -3 <= velocity_2 <= 3 and -3 <= velocity_3 <= 3:
        angle = 0

    global_coord_delta = np.array([-vxvywz[0] * time * math.sin(angle + current_angle)
                                   - vxvywz[1] * time * math.cos(angle + current_angle),
                                   -vxvywz[0] * time * math.cos(angle + current_angle)
                                   + vxvywz[1] * time * math.sin(angle + current_angle)])

    return global_coord_delta[0],  global_coord_delta[1], angle

