# test of rotation trajectory generation
#
# Z. Zhang
# 07/2025

import numpy as np

# point of center
poc = np.array([0., 0., 0.])
ang_init = 0 #np.pi / 4  # initial angle in radians
ang_rot = np.pi  # 45 degrees in radians
# check if the angle is correct: (0，180)
assert 0 <= np.abs(ang_rot) <= np.pi, "Rotation angle must be between 0 and π (0 to 180 deg)"
# init range of two grippers
D1 = 0.2  # distance between two grippers
d1 = D1/ 2  
# goal range of two grippers
D2 = 0.3  # distance between two grippers
d2 = D2 / 2
# lift height
h = 0.1  # height to lift the cloth

num_step = 100
x_ls, y_ls, z_ls = [], [], []
# generate trajectory
for i in range(num_step):
    t = i / (num_step - 1)  # normalize t to [0, 1]
    # interpolate between initial and goal positions
    dt = d1 + (d2 - d1) * t
    # current angle
    ang_rot_t = ang_init + ang_rot * t
    x = poc[0] + dt * np.cos(ang_rot_t)
    y = poc[1] + dt * np.sin(ang_rot_t)
    z = poc[2] + h * t  # lift height increases linearly
    x_ls.append(x)
    y_ls.append(y)
    z_ls.append(z)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_ls, y_ls, z_ls, marker='o', markersize=2, label='Trajectory')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Trajectory of Cloth Manipulation')
ax.set_box_aspect([1, 1, 1])  # [x, y, z]比例相同
# showcase the start points
ax.scatter(x_ls[0], y_ls[0], z_ls[0], color='red', s=50, label='Start Point')
# showcase the end points
ax.scatter(x_ls[-1], y_ls[-1], z_ls[-1], color='green', s=50, label='End Point')
ax.set_xlim(-0.20, .2)
ax.set_ylim(-.2, .2)
# top view
ax.view_init(elev=90, azim=-90)  # Set the view angle for top view
plt.legend()
plt.show()

