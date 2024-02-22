import math
import numpy as np
from keras import models
import matplotlib.pyplot as plt
import copy


def split_square_edges(step):
    square_size = 0.4

    vertices = np.array([[0, 0], [square_size, 0], [square_size, square_size], [0, square_size]])

    edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]

    points = []

    for start_idx, end_idx in edge_indices:
        start_point = vertices[start_idx]
        end_point = vertices[end_idx]

        direction = end_point - start_point
        length = np.linalg.norm(direction)

        num_points = int(length / step)

        step_vector = direction / num_points

        for i in range(num_points + 1):
            point = start_point + i * step_vector
            points.append(point)

    return np.array(points)


def calculate_omni_wheel_velocities(speed, wheel_radius=0.04):
    matrix = np.array([[-2 * math.sin(math.pi / 3), -2 * math.sin(math.pi), 2 * math.sin(math.pi / 3)],
                       [2 * math.cos(math.pi / 3), 2 * math.cos(math.pi), 2 * math.cos(math.pi / 3)],
                       [1 / 0.13, 1 / 0.13, 1 / 0.13]])
    r = wheel_radius
    X = np.array(speed)
    X1 = np.dot(1 / r, X)
    mat = np.dot(1 / 3, matrix)
    mat_inv = np.linalg.inv(mat)
    Y1 = np.matmul(mat_inv, X1)
    return np.dot(16, Y1)


def NeurophysicalModel(velocity_1, velocity_2, velocity_3, type_surface, time, current_angle):
    current_angle = current_angle
    time = time
    input_data_vel_and_type_1 = np.array([velocity_1, type_surface])
    input_data_vel_and_type_2 = np.array([velocity_2, type_surface])
    input_data_vel_and_type_3 = np.array([velocity_3, type_surface])
    input_data_vel_and_type_1 = input_data_vel_and_type_1.reshape(1, -1)
    input_data_vel_and_type_2 = input_data_vel_and_type_2.reshape(1, -1)
    input_data_vel_and_type_3 = input_data_vel_and_type_3.reshape(1, -1)
    "normalization"
    mean_input_6 = np.array([0.29462298, 1.99199573])
    std_input_6 = np.array([28.19668318, 0.81504003])
    input_data_vel_and_type_1 -= mean_input_6
    input_data_vel_and_type_1 /= std_input_6

    mean_input_10 = np.array([-0.0380413, 1.992])
    std_input_10 = np.array([28.39307565, 0.81482268])
    input_data_vel_and_type_2 = input_data_vel_and_type_2 - mean_input_10
    input_data_vel_and_type_2 /= std_input_10

    mean_input_11 = np.array([-0.29964439, 1.98727224])
    std_input_11 = np.array([28.30755824, 0.81384974])
    input_data_vel_and_type_3 -= mean_input_11
    input_data_vel_and_type_3 /= std_input_11
    "predicting current of motors"
    current_first_motor = models.load_model('current_wheel_1_all_types_first_motor_input')
    current_second_motor = models.load_model('current_wheel_2_all_types_second_motor_input')
    current_third_motor = models.load_model('current_wheel_3_all_types_third_motor_input')
    "predicting effective velocity in motors"
    effective_first_motor = models.load_model('effective_vel_1_current_1_vel_type_first_motor')
    effective_second_motor = models.load_model('effective_vel_1_current_1_vel_type_second_motor')
    effective_third_motor = models.load_model('effective_vel_1_current_1_vel_type_third_motor')

    "STRUCTURE"

    current_11 = current_first_motor.predict(input_data_vel_and_type_1)
    current_22 = current_second_motor.predict(input_data_vel_and_type_2)
    current_33 = current_third_motor.predict(input_data_vel_and_type_3)
    current_vel_type_first_motor_input = np.c_[current_11, velocity_1]
    current_vel_type_first_motor_input = np.c_[current_vel_type_first_motor_input, type_surface]

    mean_input_7 = np.array([0.62878268, 0.00487195, 1.99176503])
    std_input_7 = np.array([0.38759359, 28.54770489, 0.81499685])
    current_vel_type_first_motor_input -= mean_input_7
    current_vel_type_first_motor_input /= std_input_7
    effective_vel_111 = effective_first_motor.predict(current_vel_type_first_motor_input)
    current_vel_type_second_motor_input = np.c_[current_22, velocity_2]
    current_vel_type_second_motor_input = np.c_[current_vel_type_second_motor_input, type_surface]

    mean_input_8 = np.array([0.60241189, -0.33878786, 1.99176955])
    std_input_8 = np.array([0.37166331, 28.74375636, 0.81477325])
    current_vel_type_second_motor_input -= mean_input_8
    current_vel_type_second_motor_input /= std_input_8
    effective_vel_222 = effective_second_motor.predict(current_vel_type_second_motor_input)
    current_vel_type_third_motor_input = np.c_[current_33, velocity_3]
    current_vel_type_third_motor_input = np.c_[current_vel_type_third_motor_input, type_surface]

    mean_input_9 = np.array([0.63123638, -0.55539807, 1.99176842])
    std_input_9 = np.array([0.40148566, 28.59662026, 0.81482913])
    current_vel_type_third_motor_input -= mean_input_9
    current_vel_type_third_motor_input /= std_input_9
    effective_vel_333 = effective_third_motor.predict(current_vel_type_third_motor_input)
    if velocity_1 < 0:
        effective_vel_111[0] = -effective_vel_111[0]

    if velocity_2 < 0:
        effective_vel_222[0] = -effective_vel_222[0]

    if velocity_3 < 0:
        effective_vel_333[0] = -effective_vel_333[0]

    velocity_SHADOW = np.c_[effective_vel_111, effective_vel_222]
    velocity_SHADOW = np.c_[velocity_SHADOW, effective_vel_333]

    "Radius of omni-wheel"

    r = 0.04

    "Direct kinematics"
    matrix = np.array([[-2 * math.sin(math.pi / 3), -2 * math.sin(math.pi), 2 * math.sin(math.pi / 3)],
                       [2 * math.cos(math.pi / 3), 2 * math.cos(math.pi), 2 * math.cos(math.pi / 3)],
                       [1 / 0.13, 1 / 0.13, 1 / 0.13]])

    m1 = np.dot(1 / 3, matrix)

    mat = np.matmul(m1, velocity_SHADOW[0] / 16)

    vxvywz = np.dot(r, mat)

    dxdyda_local = np.array([vxvywz[0] * time, vxvywz[1] * time, vxvywz[2] * time])

    angle = dxdyda_local[2]
    print(angle, "дельта угол")
    if -3 <= velocity_1 <= 3 and -3 <= velocity_2 <= 3 and -3 <= velocity_3 <= 3:
        angle = 0

    global_coord_delta = np.array([-vxvywz[0] * time * math.sin(angle + current_angle)
                                   - vxvywz[1] * time * math.cos(angle + current_angle),
                                   -vxvywz[0] * time * math.cos(angle + current_angle)
                                   + vxvywz[1] * time * math.sin(angle + current_angle)])

    return dxdyda_local[0], dxdyda_local[1], dxdyda_local[2], vxvywz[0], vxvywz[1], vxvywz[2]


class PIController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.prev_error_angle = 0
        self.integral_x = 0
        self.integral_y = 0
        self.integral_angle = 0
        self.d_x = 0
        self.d_y = 0
        self.d_angle = 0

    def update(self, setpoint_x, setpoint_y, setpoint_angle, current_x, current_y, current_angle, dt):
        error_x = setpoint_x - current_x
        error_y = setpoint_y - current_y
        error_angle = setpoint_angle - current_angle

        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        self.integral_angle += error_angle * dt

        self.d_x = (error_x - self.prev_error_x)/dt
        self.d_y = (error_y - self.prev_error_y)/dt
        self.d_angle = (error_angle - self.prev_error_angle)/dt

        control_x = self.kp * error_x + self.ki * self.integral_x + self.kd * self.d_x
        control_y = self.kp * error_y + self.ki * self.integral_y + self.kd * self.d_y
        control_angle = 2*self.kp * error_angle + 2*self.ki * self.integral_angle + 2*self.kd * self.d_angle

        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_angle = error_angle

        return control_x, control_y, control_angle


kp = 6
ki = 0.8
kd = 0

controller = PIController(kp, ki, kd)

step_size = 0.02
points_array = split_square_edges(step_size)

global_coord_now = np.array([0.0, 0.0, 0.0])
global_coord_all_points = np.array([0.0, 0.0, 0.0])

i = 0
global_axis_vxvyvz = np.array([0.0, 0.0, 0.0])
while i < len(points_array):
    print("я тут")
    setpoint_x = points_array[i][0]
    setpoint_y = points_array[i][1]
    print(setpoint_x, setpoint_y, "целевая точка")
    distance = math.sqrt((global_coord_now[0] - setpoint_x) ** 2 + (global_coord_now[1] - setpoint_y) ** 2)
    while distance > 0.005:

        vx_input, vy_input, vz_input = controller.update(setpoint_x, setpoint_y, 0, global_coord_now[0], global_coord_now[1], global_coord_now[2], 0.1)
        print(vx_input, vy_input, vz_input, "скорости с регулятора")
        w1w2w3_input = calculate_omni_wheel_velocities([vx_input, vy_input, vz_input])

        print("скорости w1w2w3", w1w2w3_input)
        dxdydphi_prediction = NeurophysicalModel(w1w2w3_input[0], w1w2w3_input[1], w1w2w3_input[2], 1, 0.1, 0)
        vxvyvz = np.array([dxdydphi_prediction[3], dxdydphi_prediction[4], dxdydphi_prediction[5]])
        global_axis_vxvyvz = np.vstack((global_axis_vxvyvz, vxvyvz))
        global_coord_now[0] += dxdydphi_prediction[0]
        global_coord_now[1] += dxdydphi_prediction[1]
        global_coord_now[2] += dxdydphi_prediction[2]

        global_coord_all_points = np.vstack((global_coord_all_points, global_coord_now))
        distance = math.sqrt((global_coord_now[0] - setpoint_x)**2 + (global_coord_now[1] - setpoint_y)**2)
        print("достигаю точку:", i)
        print("целевая точка", setpoint_x, setpoint_y)
        print("global_coord_all_points", global_coord_now)

    i += 1

angle_shadow = copy.deepcopy(global_coord_all_points)

global_coord_delta_shadow = np.array([-global_axis_vxvyvz[0][0]*0.1 * math.sin(angle_shadow[0][2]) - global_axis_vxvyvz[0][1]*0.1 * math.cos(angle_shadow[0][2]),
                                   -global_axis_vxvyvz[0][0]*0.1 * math.cos(angle_shadow[0][2]) + global_axis_vxvyvz[0][1]*0.1 * math.sin(angle_shadow[0][2])])

for j in range(1, len(global_axis_vxvyvz[:, 0])):
    new_line_delta_global4 = np.array([-global_axis_vxvyvz[j][0]*0.1 * math.sin(angle_shadow[j][2]) - global_axis_vxvyvz[j][1]*0.1 * math.cos(angle_shadow[j][2]),
                                   -global_axis_vxvyvz[j][0]*0.1 * math.cos(angle_shadow[j][2]) + global_axis_vxvyvz[j][1]*0.1 * math.sin(angle_shadow[j][2])])
    global_coord_delta_shadow = np.vstack((global_coord_delta_shadow, new_line_delta_global4))

track_shadow = copy.deepcopy(global_coord_delta_shadow)

for j in range(1, len(global_coord_delta_shadow[:, 0])):
    track_shadow[j] = track_shadow[j - 1] + global_coord_delta_shadow[j]


plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = '32'
plt.plot(track_shadow[:, 0], track_shadow[:, 1], 'r-', linewidth=3)
# plt.plot(points_array[:, 0], points_array[:, 1], 'g-', linewidth=3)
# plt.plot(range(0, len(global_coord_all_points[:, 2])), global_coord_all_points[:, 2], 'r-', linewidth=3)
plt.xlabel('X axis (m.)', fontsize=40)
plt.ylabel('Y axis (m.)', fontsize=40)
plt.title('Trajectory prediction considering PI controller')
plt.legend()
plt.grid(True, color="grey", linewidth="0.8", linestyle="-")
plt.show()

plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = '32'
plt.plot(range(0, len(global_coord_all_points[:, 2])), global_coord_all_points[:, 2], 'r-', linewidth=3)
plt.xlabel('Control action number', fontsize=40)
plt.ylabel('Angle error (rad.)', fontsize=40)
plt.title('Angle error')
plt.legend()
plt.grid(True, color="grey", linewidth="0.8", linestyle="-")
plt.show()
