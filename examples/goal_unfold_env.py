# copied from random_env.py
#
# Z. Zhang
# 10/2024

import os.path as osp
import argparse
import numpy as np
import os

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

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
def calculate_values_geodesic(all_pts, key_indices, geodesic_distances):
    # cal. the minimum geodesic distance from each point in all_pts to the key points specified by key_indices.
    min_distances = np.min(geodesic_distances[:, key_indices], axis=1)
    
    # to normalize the values so they fall between 0 and 1.
    max_distance = np.max(min_distances) if np.max(min_distances) > 0 else 1
    # Points closer to the key points will have higher values, while points further away will have lower values.
    values = 1 - (min_distances / max_distance)    
    return values

# Function to return posi of key points
def get_goal_posi(key_indices):
    all_points = pyflex.get_positions().reshape(-1, 4)

    # # get values of all particles on cloth
    # # Compute geodesic distances
    # geodesic_distances = compute_geodesic_distance(all_points[:,:3])

    # # Calculate values based on geodesic distances
    # values = calculate_values_geodesic(all_points, key_indices, geodesic_distances)
    # TODO: get point positions according to the max value from top view, rather than the key point positions

    # get the positions of the key points
    key_point_pos = all_points[key_indices, :3]
    
    # randomly select 1 key point as the goal point
    return key_point_pos[np.random.choice(key_point_pos.shape[0], 1)]


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop',
    # 'ClothFlattenPPP']
    parser.add_argument('--env_name', type=str, default='ClothFlattenPPP')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

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
    for i in range(env.horizon):
        index_order = index_temp[i%2]
        # action = env.action_space.sample()

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
        goal_posi = get_goal_posi(grasp_key)

        # specify unp(ick) and p(ick) for two pickers 

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
