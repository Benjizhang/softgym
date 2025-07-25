# test of rotation trajectory generation
#
# Z. Zhang
# 07/2025

import numpy as np
from numpy.linalg import norm

def get_line_angle(p1, p2, p3, p4):
    """
    使用numpy现成函数计算两条直线的夹角
    
    参数:
    p1, p2: 第一条直线上的两个点
    p3, p4: 第二条直线上的两个点
    
    返回:
    夹角（度，取0-90度之间的最小角）
    """
    # 计算方向向量（使用numpy的数组减法）
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)
    
    # 检查向量是否有效
    if norm(v1) == 0 or norm(v2) == 0:
        raise ValueError("两点不能重合以确定直线")
    
    # 计算单位向量的点积（使用现成的点积函数）
    unit_v1 = v1 / norm(v1)
    unit_v2 = v2 / norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    
    # 计算弧度（使用numpy的反余弦函数）并转换为角度（使用numpy的角度转换函数）
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # 返回最小夹角
    return min(angle_deg, 180 - angle_deg)

# point of center
poc = np.array([0., 0., 0.])
ang_init = 0 #np.pi / 4  # initial angle in radians
ang_init = np.round(np.random.uniform(-np.pi, np.pi),2)
ang_rot = np.round(np.random.uniform(0, np.pi),2)
print(f"Initial angle: {np.round(np.rad2deg(ang_init),2)} deg, Rotation angle: {np.round(np.rad2deg(ang_rot),2)} deg")

assert -np.pi <= ang_init <= np.pi, "Initial angle must be between -π and π (-180 to 180 deg)"
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
x1_ls, y1_ls, z1_ls = [], [], []
x2_ls, y2_ls, z2_ls = [], [], []
# generate trajectory
for i in range(num_step):
    t = i / (num_step - 1)  # normalize t to [0, 1]
    # interpolate between initial and goal positions
    dt = d1 + (d2 - d1) * t
        
    # x = poc[0] + dt * np.cos(ang_rot_t)
    # y = poc[1] + dt * np.sin(ang_rot_t)
    # z = poc[2] + h * t  # lift height increases linearly
    # x_ls.append(x)
    # y_ls.append(y)
    # z_ls.append(z)

    # first gripper
    ang_rot_t = ang_init + ang_rot * t
    x1 = poc[0] + dt * np.cos(ang_rot_t)
    y1 = poc[1] + dt * np.sin(ang_rot_t)
    z1 = poc[2] + h * t  # lift height increases linearly
    x1_ls.append(x1)
    y1_ls.append(y1)
    z1_ls.append(z1)
    # second gripper
    ang2_rot_t = (ang_init + np.pi) + ang_rot * t
    x2 = poc[0] + dt * np.cos(ang2_rot_t)
    y2 = poc[1] + dt * np.sin(ang2_rot_t)
    z2 = poc[2] + h * t  # lift height increases linearly
    x2_ls.append(x2)
    y2_ls.append(y2)
    z2_ls.append(z2)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1_ls, y1_ls, z1_ls, marker='o', markersize=2, label='Traj1')
ax.plot(x2_ls, y2_ls, z2_ls, marker='o', markersize=2, label='Traj2')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Trajectory of Cloth Manipulation')
# ax.set_box_aspect([1, 1, 1])  # [x, y, z]
# showcase the start points
ax.scatter(x1_ls[0], y1_ls[0], z1_ls[0], color='red', s=50, label='Start Point')
ax.scatter(x1_ls[-1], y1_ls[-1], z1_ls[-1], color='green', s=50, label='End Point')
ax.scatter(x2_ls[0], y2_ls[0], z2_ls[0], color='red', s=50,)
ax.scatter(x2_ls[-1], y2_ls[-1], z2_ls[-1], color='green', s=50,)

# draw a dashed line between two start points
ax.plot([x1_ls[0], x2_ls[0]], [y1_ls[0], y2_ls[0]], [z1_ls[0], z2_ls[0]], color='black', linestyle='--', linewidth=1)
# draw a dashed line between two end points
ax.plot([x1_ls[-1], x2_ls[-1]], [y1_ls[-1], y2_ls[-1]], [z1_ls[-1], z2_ls[-1]], color='blue', linestyle='--', linewidth=1)
# calculate the angle between the above two dased lines
line_angle = get_line_angle(
    (x1_ls[0], y1_ls[0], z1_ls[0]), (x2_ls[0], y2_ls[0], z2_ls[0]),
    (x1_ls[-1], y1_ls[-1], z1_ls[-1]), (x2_ls[-1], y2_ls[-1], z2_ls[-1])
)
print(f"Angle between the two dashed lines: {line_angle:.2f} degrees")
# calculate the length of the two dashed lines
line1_length = norm(np.array([x1_ls[0], y1_ls[0], z1_ls[0]]) - np.array([x2_ls[0], y2_ls[0], z2_ls[0]]))
line2_length = norm(np.array([x1_ls[-1], y1_ls[-1], z1_ls[-1]]) - np.array([x2_ls[-1], y2_ls[-1], z2_ls[-1]]))
print(f"Length of the first dashed line: {line1_length:.2f}, second dashed line: {line2_length:.2f}")


ax.set_xlim(-0.20, .2)
ax.set_ylim(-.2, .2)
# top view
ax.view_init(elev=90, azim=-90)  # Set the view angle for top view
plt.legend()
plt.show()

