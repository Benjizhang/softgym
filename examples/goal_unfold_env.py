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


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop',
    # 'ClothFlattenPPP']
    parser.add_argument('--env_name', type=str, default='ClothFlatten')
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

    center_pose = np.array([0.0, 0.2, 0.0])
    # define rest posi for two pickers
    rest_array = np.array([[*center_pose[:2], 0.5], [*center_pose[:2], -0.5]])
    # define display posi (hanging cloth and wait stable) for two pickers
    disply_pose = np.tile(center_pose, (2, 1))
    
    frames = [env.get_image(args.img_size, args.img_size)]
    for i in range(env.horizon):
        action = env.action_space.sample()
        # determine the grasp point on the cloth as goal of picker

        # specify unp(ick) and p(ick) for two pickers

        # unp: [goal, 1]; p: [current, 1]

        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])
        
        # after get to the goal
        # change unp and p for two pickers
        
        # unp: [rest, 0]; p: [display, 1]

        if args.test_depth:
            show_depth()

    if args.save_video_dir is not None:
        os.makedirs(args.save_video_dir, exist_ok=True)
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()
