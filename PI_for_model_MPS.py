import math
import numpy as np
from keras import models
import matplotlib.pyplot as plt
import copy

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

    if -3 <= velocity_1 <= 3 and -3 <= velocity_2 <= 3 and -3 <= velocity_3 <= 3:
        angle = 0

    global_coord_delta = np.array([-vxvywz[0] * time * math.sin(angle + current_angle)
                                   - vxvywz[1] * time * math.cos(angle + current_angle),
                                   -vxvywz[0] * time * math.cos(angle + current_angle)
                                   + vxvywz[1] * time * math.sin(angle + current_angle)])

    return velocity_SHADOW[0][0], velocity_SHADOW[0][1], velocity_SHADOW[0][2]


class PIController:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.integral = 0

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        output = self.kp * error + self.ki * self.integral
        return output


def velocity_with_pi_controller(vx, vy, vz, type_surface):
    w1w2w3_input = calculate_omni_wheel_velocities([vx, vy, vz])
    w1w2w3_prediction = NeurophysicalModel(w1w2w3_input[0], w1w2w3_input[1], w1w2w3_input[2], type_surface, 0.5, 0)

    kp1 = 1.9
    ki1 = 1.2
    kp2 = 1.9
    ki2 = 1.2
    kp3 = 1.9
    ki3 = 1.2

    w1_setpoint = w1w2w3_input[0]
    w1_measured_value = w1w2w3_prediction[0]
    w2_setpoint = w1w2w3_input[1]
    w2_measured_value = w1w2w3_prediction[1]
    w3_setpoint = w1w2w3_input[2]
    w3_measured_value = w1w2w3_prediction[2]

    controller_w1 = PIController(kp1, ki1)
    controller_w2 = PIController(kp2, ki2)
    controller_w3 = PIController(kp3, ki3)

    dt = 0.1
    total_time = 12
    time = 0

    time_values = []
    measured_values_w1 = []
    measured_values_w2 = []
    measured_values_w3 = []

    output_1 = 0
    output_2 = 0
    output_3 = 0
    while time < total_time:

        output_1 = controller_w1.update(w1_setpoint, w1_measured_value, dt)

        w1_measured_value_temp = NeurophysicalModel(output_1, w1w2w3_input[1], w1w2w3_input[2], type_surface, 0.5, 0)
        w1_measured_value = w1_measured_value_temp[0]
        print(w1_setpoint, "уставка w1")
        print(w1_measured_value, "измеренное w1")
        print(output_1, "output_1")
        measured_values_w1.append(w1_measured_value_temp[0])


        if -3.5 <= w2_setpoint <= 3.5:
            output_2 = 0
        else:
            output_2 = controller_w2.update(w2_setpoint, w2_measured_value, dt)
            w2_measured_value_temp = NeurophysicalModel(w1w2w3_input[0], output_2, w1w2w3_input[2], type_surface, 0.5,
                                                        0)
            w2_measured_value = w2_measured_value_temp[1]
            print(w2_setpoint, "уставка w2")
            print(w2_measured_value, "измеренное w2")
            print(output_2, "output_2")
            measured_values_w2.append(w2_measured_value_temp[1])

        output_3 = controller_w3.update(w3_setpoint, w3_measured_value, dt)

        w3_measured_value_temp = NeurophysicalModel(w1w2w3_input[0], w1w2w3_input[1], output_3, type_surface, 0.5, 0)
        w3_measured_value = w3_measured_value_temp[2]
        print(w3_setpoint, "уставка w3")
        print(w3_measured_value, "измеренное w3")
        print(output_3, "output_3")
        measured_values_w3.append(w3_measured_value_temp[2])

        time_values.append(time)
        time += dt

    control_output_in_model_tmp = NeurophysicalModel(output_1, output_2, output_3, 1, 0.5, 0)

    _real_velocity = np.array([control_output_in_model_tmp[0], control_output_in_model_tmp[1], control_output_in_model_tmp[2]])

    return _real_velocity


def direct_kinematics(velocity_w1w2w3):

    real_velocity = np.array(velocity_w1w2w3)
    "Radius of omni-wheel"

    r = 0.04

    "Direct kinematics"
    matrix = np.array([[-2 * math.sin(math.pi / 3), -2 * math.sin(math.pi), 2 * math.sin(math.pi / 3)],
                       [2 * math.cos(math.pi / 3), 2 * math.cos(math.pi), 2 * math.cos(math.pi / 3)],
                       [1 / 0.13, 1 / 0.13, 1 / 0.13]])

    m1 = np.dot(1 / 3, matrix)

    mat_model = np.matmul(m1, real_velocity / 16)
    vxvywz_model = np.dot(r, mat_model)

    dxdyda_local_model = np.array([vxvywz_model[0] * 5, vxvywz_model[1] * 5,
                                   vxvywz_model[2] * 5])

    angle_model = dxdyda_local_model[2]

    global_coord_delta_model = np.array([-vxvywz_model[0] * 5 * math.sin(angle_model) -
                                         vxvywz_model[1] * 5 * math.cos(angle_model),
                                         -vxvywz_model[0] * 5 * math.cos(angle_model) +
                                         vxvywz_model[1] * 5 * math.sin(angle_model)])

    print(global_coord_delta_model)

    return global_coord_delta_model


if __name__ == "__main__":

    real_velocity = [[-0.025, 0, 0, 1], [0, 0.025, 0, 1], [0.025, 0, 0, 1], [0, -0.025, 0, 1]]

    input_velocity = [[0] * 4] * 4
    input_velocity_target = [[0] * 4] * 4
    for i in range(len(input_velocity)):
        vx = real_velocity[i][0]
        vy = real_velocity[i][1]
        vz = real_velocity[i][2]
        type = real_velocity[i][3]
        input_velocity[i] = velocity_with_pi_controller(vx, vy, vz, type)
        input_velocity_target[i] = calculate_omni_wheel_velocities([vx, vy, vz])

    print(input_velocity, "конечные уставные скорости робота")
    print(input_velocity_target, "целевые скорости робота")
    global_coord_dxdydz = np.array([0, 0])
    global_coord_dxdydz_real = np.array([0, 0])
    for i in range(0, len(input_velocity)):
        new_line1 = direct_kinematics(input_velocity[i])
        global_coord_dxdydz = np.vstack((global_coord_dxdydz, new_line1))

        new_line2 = direct_kinematics(input_velocity_target[i])
        global_coord_dxdydz_real = np.vstack((global_coord_dxdydz_real, new_line2))

    track_model = copy.deepcopy(global_coord_dxdydz)
    track_real = copy.deepcopy(global_coord_dxdydz_real)
    for j in range(0, 5):
        track_model[j] = track_model[j - 1] + global_coord_dxdydz[j]
        track_real[j] = track_real[j - 1] + global_coord_dxdydz_real[j]

    # plt.plot(time_values, measured_values_w1, 'r-', label='Measured Value_w1')
    # plt.plot(time_values, measured_values_w2, 'b-', label='Measured Value_w2')
    # plt.plot(time_values, measured_values_w3, 'g-', label='Measured Value_w3')
    # plt.plot(time_values, [w1_setpoint] * len(time_values), linestyle='--', label='Setpoint')
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.size'] = '32'
    plt.plot(track_model[:, 0], track_model[:, 1], 'r-', linewidth=3)
    plt.plot(track_real[:, 0], track_real[:, 1], 'g-', linewidth=3)
    plt.xlabel('X axis (m.)', fontsize=40)
    plt.ylabel('Y axis (m.)', fontsize=40)
    plt.title('Trajectory prediction considering PI controller')
    plt.legend()
    plt.grid(True, color="grey", linewidth="0.8", linestyle="-")
    plt.show()
