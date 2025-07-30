# ::goal_unfold_env2.py
# def: particle -> on cloth, point -> point cloud
# rotating cloth to unfold it
# 
# Z. Zhang
# 7/2025

import os.path as osp
import argparse
import numpy as np
import os
import scipy
import math

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from softgym.utils.camera_utils import get_world_coords,get_rgbd_and_mask

def show_depth():
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    plt.show()

# Function to compute geodesic distances
def compute_geodesic_distance(points):
    assert points.shape[1] == 3
    k = 5  # Number of neighbors
    # a sparse adjacency matrix representing a graph where each point is connected to its k nearest neighbors.
    adjacency_matrix = kneighbors_graph(points, k, mode='distance', include_self=False).toarray()
    # the shortest paths in the graph. The method used is Dijkstra's algorithm
    geodesic_distances = shortest_path(adjacency_matrix, method='D', directed=False) # 2D array
    return geodesic_distances

# Function to calculate values based on geodesic distances
def calculate_values_geodesic(key_indices, geodesic_distances):
    # cal. the minimum geodesic distance from each point in all_pts to the key points specified by key_indices.
    min_distances = np.min(geodesic_distances[:, key_indices], axis=1)
    
    # to normalize the values so they fall between 0 and 1.
    max_distance = np.max(min_distances) if np.max(min_distances) > 0 else 1
    # Points closer to the key points will have higher values, while points further away will have lower values.
    values = 1 - (min_distances / max_distance)    
    return values

# randomly return position of one key point
def random_one_key_pose(key_indices):
    all_points = pyflex.get_positions().reshape(-1, 4)

    # # get values of all particles on cloth
    # # Compute geodesic distances
    # geodesic_distances = compute_geodesic_distance(all_points[:,:3])

    # # Calculate values based on geodesic distances
    # values = calculate_values_geodesic(key_indices, geodesic_distances)

    # get the positions of the key points
    key_point_pos = all_points[key_indices, :3]
    
    # randomly select 1 key point as the goal point
    return key_point_pos[np.random.choice(key_point_pos.shape[0], 1)]

# Function to get observable particle index
def get_observable_particle_index_old(world_coords, particle_pos, rgb, depth):
    height, width, _ = rgb.shape
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]
    # this is just pointcloud positions
    estimated_world_coords = np.array(world_coords)[np.where(depth.flatten() > 0)][:, :3]

    distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)

    estimated_particle_idx = np.argmin(distance, axis=1) # min index for each row (find cloth particle index for each pixel)

    estimated_particle_idx = np.unique(estimated_particle_idx) # index of pts on cloth
    assert len(estimated_particle_idx) < particle_pos.shape[0]
    return np.array(estimated_particle_idx, dtype=np.int32)

# get values of point clouds
def get_ptcloud_value(key_indices, ptclouds, particle_pos):
    assert ptclouds.shape[1] == 3
    assert particle_pos.shape[1] == 3
    distance = scipy.spatial.distance.cdist(ptclouds, particle_pos)
    # cloth particle index for each point cloud
    estimated_particle_idx = np.argmin(distance, axis=1) # min index for each row (find cloth particle index for each pixel)
    estimated_particle_idx = np.array(estimated_particle_idx, dtype=np.int32)
    ptcloud_values = get_particle_value_from_idx(key_indices,estimated_particle_idx)
    return ptcloud_values

# plot ptcloud values
def plot_ptcloud_value(key_indices,env):
    # get particle positions
    particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
    # get ptcloud
    ptclouds, _ = get_pointcloud_and_idx_of_cloth(env) 
    # get values of point clouds
    ptcloud_values = get_ptcloud_value(key_indices, ptclouds, particle_pos)
    assert ptclouds.shape[0] == ptcloud_values.shape[0]
    # plot the color map
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with color based on values
    scatter = ax.scatter(ptclouds[:, 2], ptclouds[:, 0], ptclouds[:, 1], c=ptcloud_values, cmap='viridis', s=10)
    plt.colorbar(scatter, label='Value based on Geodesic Distance')
    ax.scatter(particle_pos[key_indices, 2], particle_pos[key_indices, 0], particle_pos[key_indices, 1], color='red', label='Key Points', edgecolor='black', s=100)
    ax.set_title('Pt Cloud Value to Key Features (w/ Geodesic Distances)')
    ax.set_xlabel('Z'),ax.set_ylabel('X'),ax.set_zlabel('Y')
    ax.view_init(elev=74, azim=90) # view angle ROUGHLY same as the camera
    plt.legend()
    plt.show()

# get point clouds and corresponding indices of partilces on cloth
def get_pointcloud_and_idx_of_cloth(env):
    cloth_mask, rgb, depth = get_rgbd_and_mask(env, 0)
    world_coordinates = get_world_coords(rgb, depth, env)[:, :, :3].reshape((-1, 3)) # based on image frame e.g. 720x720
    pointcloud = world_coordinates[depth.flatten() > 0].astype(np.float32) # based on image frame e.g. num < 720x720

    position = pyflex.get_positions().reshape(-1, 4)[:, :3]
    observable_idx = get_observable_particle_index_old(world_coordinates, position, rgb, depth) # indices of pts on cloth

    return pointcloud, observable_idx

# get gt values of all particles on the cloth
def get_all_particle_value(key_indices):
    all_points = pyflex.get_positions().reshape(-1, 4)
    geodesic_distances = compute_geodesic_distance(all_points[:,:3])
    all_particle_values = calculate_values_geodesic(key_indices, geodesic_distances) 
    return all_particle_values

# get values of observable particles on the cloth (given indices)
def get_particle_value_from_idx(key_indices, observable_idx):
    all_points_values = get_all_particle_value(key_indices)
    pointcloud_values = all_points_values[observable_idx] # observable_idx: must be indices of pts on cloth
    return pointcloud_values

# func to get grasp positions from point clouds
# def get_grasp_posi(policy_name, key_indices, env):


def plot_particle_value_from_ptcloud(key_indices, env):
    _, observable_idx = get_pointcloud_and_idx_of_cloth(env) # observable_idx: indices of pts on cloth
    obs_cloth_pts_value = get_particle_value_from_idx(key_indices, observable_idx)
    
    all_cloth_pts = pyflex.get_positions().reshape(-1, 4)
    obs_cloth_pts = all_cloth_pts[observable_idx]
    # plot the color map
    # Visualization in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color based on values
    scatter = ax.scatter(obs_cloth_pts[:, 2], obs_cloth_pts[:, 0], obs_cloth_pts[:, 1], c=obs_cloth_pts_value, cmap='viridis', s=10)
    plt.colorbar(scatter, label='Value based on Geodesic Distance')
    ax.scatter(all_cloth_pts[key_indices, 2], all_cloth_pts[key_indices, 0], all_cloth_pts[key_indices, 1], color='red', label='Key Points', edgecolor='black', s=100)
    ax.set_title('Cloth Value to Key Features (w/ Geodesic Distances)')
    ax.set_xlabel('Z'),ax.set_ylabel('X'),ax.set_zlabel('Y')
    ax.view_init(elev=74, azim=90) # view angle ROUGHLY same as the camera
    plt.legend()
    plt.show()

# Function to generate rotation trajectories for two grippers
def gen_rotation_traj(traj_para):
    num_step = traj_para['num_step'] if traj_para['num_step'] is not None else 100
    d1 = traj_para['d1']
    d2 = traj_para['d2']
    poc = traj_para['poc']
    assert poc.shape == (3,), "POC must be a 3D point (x, y, z)"
    ang_init = traj_para['ang_init']
    ang_rot = traj_para['ang_rot']
    h = traj_para['height_lift']  # lift height

    x1_ls, y1_ls, z1_ls, x2_ls, y2_ls, z2_ls = [], [], [], [], [], []
    for i in range(num_step):
        t = i / (num_step - 1)  # normalize t to [0, 1]
        # interpolate between initial and goal positions
        dt = d1 + (d2 - d1) * t
            
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
    # Convert lists to numpy arrays
    x1_ls = np.array(x1_ls)
    y1_ls = np.array(y1_ls)
    z1_ls = np.array(z1_ls)
    x2_ls = np.array(x2_ls)
    y2_ls = np.array(y2_ls)
    z2_ls = np.array(z2_ls)
    # Return the trajectory points for both grippers
    traj1 = np.column_stack((x1_ls, y1_ls, z1_ls))  # Trajectory for the first gripper
    traj2 = np.column_stack((x2_ls, y2_ls, z2_ls))  # Trajectory for the second gripper
    return traj1, traj2 

# Function to find the best point of center (POC) based on given distance and angle
def find_best_poc(cloth_pt, ang_init, d1):
    assert np.round(d1-0,2)>1e-4, "d1 must be larger than 0"
    all_points_2d = cloth_pt[:, :2]  # Extract 2D coordinates (x, y) from the cloth points
    min_error = float('inf')
    best_pair = None
    best_poc = None
    
    num_points = len(all_points_2d)
    # Iterate through all pairs of points to find the best pair
    for i in range(num_points):
        p1 = np.array(all_points_2d[i])
        for j in range(i + 1, num_points):
            p2 = np.array(all_points_2d[j])
            # distance between p1 and p2
            distance = np.linalg.norm(p2 - p1)
            distance_error = abs(distance - d1) / d1 if d1 != 0 else abs(distance)
            # angle of line p1-p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue            
            angle = math.atan2(dy, dx)
            angle_diff = abs(angle - ang_init)
            angle_error = min(angle_diff, 2*math.pi - angle_diff) / (2*math.pi)  # 归一化到[0, 0.5]
            # total error is a combination of distance and angle errors
            total_error = 0.5 * distance_error + 0.5 * angle_error
            if total_error < min_error:
                min_error = total_error
                best_pair = (p1, p2)
                best_pair_index = (i, j)
                best_poc = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    if best_poc is None:
        raise ValueError("No valid point pairs were found.")
    poc_3d = np.array(best_poc+(0,))
    return poc_3d, best_pair, best_pair_index, min_error

def get_rotation_traj(traj_para):
    # cloth points (x,y,z,1/mass)
    all_points = pyflex.get_positions().reshape(-1, 4)
    # find best POC
    poc, best_pair, best_pair_index, min_error = find_best_poc(all_points, traj_para['ang_init'], traj_para['d1'])
    traj_para['poc'] = poc
    # generate trajectory points
    traj1, traj2 = gen_rotation_traj(traj_para)
    rot_traj = [traj1, traj2]
    return rot_traj, traj_para


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop',
    # 'ClothFlattenPPP','TshirtFlatten']
    parser.add_argument('--env_name', type=str, default='ClothFlattenPPP')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./video/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')

    args = parser.parse_args() # get parameters from command line

    env_kwargs = env_arg_dict[args.env_name] # get parameters from the given env_arg_dict (kwarg: keyword arguments)
    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    target_env = SOFTGYM_ENVS[args.env_name](**env_kwargs)
    env = normalize(target_env) # normalize the env
    env.reset()
    
    # show_depth() # to test the depth rendering
    key_indices = env._wrapped_env._get_key_point_idx()[[0,2]]
    # plot_particle_value_from_ptcloud(key_indices,env._wrapped_env)
    # plot_ptcloud_value(key_indices,env._wrapped_env)

    center_pose = np.array([0.0, 0.5, 0.0])
    # define rest posi for two pickers
    rest_array = np.array([[center_pose[0],0.5, 0.3], [center_pose[0],0.5, -0.3]])
    # stretch_pose = np.array([[center_pose[0],0.5, 0.368/4], [center_pose[0],0.5, -0.368/4]])
    stretch_pose = np.array([[center_pose[0],0.5, 0.6/4], [center_pose[0],0.5, -0.6/4]])

    # define display posi (hanging cloth and wait stable) for two pickers
    disply_pose = np.tile(center_pose, (2, 1))
    
    frames = [env.get_image(args.img_size, args.img_size)]
    key_indices = env._wrapped_env._get_key_point_idx()[[0,2]]
    index_temp = np.array([[0,1],[1,0]])
    grasp_key_prev = -1
    stretch_flag = 0
    traj_para = {
        'num_step': 100,
        'D1': 0.2,  # initial distance between two grippers
        'D2': 0.3,  # goal distance between two grippers
        'd1': None, 
        'd2': None, 
        'poc': None,  # point of center
        'ang_init': np.random.uniform(-np.pi, np.pi),  # initial angle in radians
        'ang_rot': np.random.uniform(0, np.pi),  # rotation angle in radians
        'height_lift': 0.1  # height to lift the cloth
    }
    traj_para['d1'] = traj_para['D1'] / 2  # initial distance between two grippers
    traj_para['d2'] = traj_para['D2'] / 2  # goal distance between two grippers

    for i in range(env.horizon):
        index_order = index_temp[i%2]
        # action = env.action_space.sample()

        # step1: determine picking pts on the cloth & rotation traj for two pickers
        rot_traj, traj_para = get_rotation_traj(traj_para)

        # step3: execute the action

        # step4: execute the rotation action

        # step5: unpick the cloth after rotation

        # initialize the action for two pickers
        action = np.zeros((2, 4))
        assert action.shape == (2, 4)
        # determine the grasp point on the cloth as goal of picker
        grasp_key = key_indices[np.random.choice(key_indices.shape[0], 1)]
        if grasp_key == grasp_key_prev and i > 0:
            while grasp_key == grasp_key_prev:
                grasp_key = key_indices[np.random.choice(key_indices.shape[0], 1 )]
            stretch_flag = 1
        grasp_key_prev = grasp_key
        goal_posi = random_one_key_pose(grasp_key)

        # specify unp(ick)/0 and p(ick)/1 for two pickers
        # unp: [goal, 1]; p: [current, 1]
        unp_action = np.hstack((goal_posi[0], 1)) 
        action[index_order[0]] = unp_action
        picker_posi = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
        assert picker_posi.shape == (2, 3)
        p_action = np.hstack((picker_posi[index_order[1]], 1))
        action[index_order[1]] = p_action
        action = action.flatten()

        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])
        
        if stretch_flag:
            # stretch the cloth for rest_array position
            action = np.hstack((stretch_pose, np.array([[1],[1]])))
            action = action.flatten()

            _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
            frames.extend(info['flex_env_recorded_frames'])
            stretch_flag = 0
            break

        # after get to the goal
        action = np.zeros((2, 4))
        # unp -> p, p -> unp       
        # p: [display, 1]; unp: [rest, 0]
        p_action = np.hstack((disply_pose[index_order[0]], 1))
        unp_action = np.hstack((rest_array[index_order[1]], 0))
        action[index_order[0]] = p_action
        action[index_order[1]] = unp_action
        action = action.flatten()

        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])

        if args.test_depth:
            show_depth()

    if args.save_video_dir is not None:
        os.makedirs(args.save_video_dir, exist_ok=True)
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()
