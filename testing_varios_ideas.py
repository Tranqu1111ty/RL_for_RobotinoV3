import numpy as np
import matplotlib.pyplot as plt

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


step_size = 0.02
points_array = split_square_edges(step_size)
print(len(points_array))
global_coord_now = np.array([0.0, 1.0, 0.0])
print(global_coord_now[1])

plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = '32'
plt.plot(range(0, len(points_array[:, 1])), points_array[:, 1], 'g-', linewidth=3)
plt.xlabel('X axis (m.)', fontsize=40)
plt.ylabel('Y axis (m.)', fontsize=40)
plt.title('Trajectory prediction considering PI controller')
plt.legend()
plt.grid(True, color="grey", linewidth="0.8", linestyle="-")
plt.show()
plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = '32'
plt.plot(range(0, len(points_array[:, 1])), points_array[:, 1], 'g-', linewidth=3)
plt.xlabel('X axis (m.)', fontsize=40)
plt.ylabel('Y axis (m.)', fontsize=40)
plt.title('Trajectory prediction considering PI controller')
plt.legend()
plt.grid(True, color="grey", linewidth="0.8", linestyle="-")
plt.show()
