import os.path as osp
import numpy as np
from tqdm import tqdm
import argparse
import h5py

from sim_env import SimulationEnv
from utils import mkdir, project_pts_to_3d
from fusion import TSDFVolume

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, help='path to data')
parser.add_argument('--train_num', type=int, help='number of training sequences')
parser.add_argument('--test_num', type=int, help='number of testing sequences')
parser.add_argument('--object_type', type=str, default='ycb', choices=['cube', 'shapenet', 'ycb'])
parser.add_argument('--max_path_length', type=int, default=10, help='maximum length for each sequence')
parser.add_argument('--object_num', type=int, default=4, help='number of objects')
parser.add_argument('--start_at', type=int, default=0, help='start generating at element')
parser.add_argument('--part_size', type=int, default=4, help='size of the part per run')

'''
array([[ 1.        , -0.        , -0.        ,  0.5       ],
       [-0.        , -0.3939193 ,  0.919145  , -0.69999998],
       [ 0.        , -0.919145  , -0.3939193 ,  0.3       ],
       [ 0.        , -0.        , -0.        ,  1.        ]])


array([[924.27737975,   0.        , 640.        ],
       [  0.        , 924.27737975, 480.        ],
       [  0.        ,   0.        ,   1.        ]])


array([[ 1.        , -0.        , -0.        ,  0.5       ],
       [-0.        , -0.3939193 ,  0.919145  , -0.69999998],
       [ 0.        , -0.919145  , -0.3939193 ,  0.3       ],
       [ 0.        , -0.        , -0.        ,  1.        ]])
       
       

array([[231.06934494,   0.        , 160.        ],
       [  0.        , 231.06934494, 120.        ],
       [  0.        ,   0.        ,   1.        ]])
       
'''

def main():

    args = parser.parse_args()

    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))
    mkdir(args.data_path, clean=False)

    env = SimulationEnv(gui_enabled=False)
    camera_pose = env.sim.camera_params[0]['camera_pose']
    camera_intr = env.sim.camera_params[0]['camera_intr']
    camera_pose_small = env.sim.camera_params[1]['camera_pose']
    camera_intr_small = env.sim.camera_params[1]['camera_intr']

    part_counter = 0
    for rollout in tqdm(range(args.start_at, args.train_num + args.test_num)):

        if part_counter == args.part_size:
            print(f'Stopping here because we reached the part size: {args.part_size}.')
            return

        env.reset(args.object_num, args.object_type)
        for step_num in range(args.max_path_length):
            f = h5py.File(osp.join(args.data_path, '%d_%d.hdf5' % (rollout, step_num)), 'w')

            output = env.poke()
            for key, val in output.items():
                if key == 'action':
                    action = val
                    f['action'] = np.array([action['0'], action['1'], action['2']])
                elif key == 'camera_params_small':
                    # ignores the camera parameters
                    pass
                else:
                    f.create_dataset(key, data=val, compression="gzip", compression_opts=4)

            # tsdf
            tsdf = get_volume(
                color_image=output['color_image'],
                depth_image=output['depth_image'],
                cam_intr=camera_intr,
                cam_pose=camera_pose
            )
            f.create_dataset('tsdf', data=tsdf, compression="gzip", compression_opts=4)

            # 3d pts
            color_image_small = output['color_image_small']
            depth_image_small = output['depth_image_small']
            camera_params_small = output['camera_params_small']
            pts_small = [
                project_pts_to_3d(color_image_small[i], depth_image_small[i], camera_params_small[i]['camera_intr'], camera_params_small[i]['camera_pose'])
                for i in range(color_image_small.shape[0])
            ]
            f.create_dataset('pts_small', data=pts_small, compression="gzip", compression_opts=4)

            if step_num == args.max_path_length - 1:
                g_next = f.create_group('next')
                output = env.get_scene_info(mask_info=True)
                for key, val in output.items():
                    # ignores the camera parameters
                    if key == 'camera_params_small':
                        continue
                    g_next.create_dataset(key, data=val, compression="gzip", compression_opts=4)
            f.close()

        # Counts an element as processed
        part_counter += 1
    
    id_list = [i for i in range(args.train_num + args.test_num)]
    np.random.shuffle(id_list)

    with open(osp.join(args.data_path, 'train.txt'), 'w') as f:
        for k in range(args.train_num):
            print(id_list[k], file=f)

    with open(osp.join(args.data_path, 'test.txt'), 'w') as f:
        for k in range(args.train_num, args.train_num + args.test_num):
            print(id_list[k], file=f)


def get_volume(color_image, depth_image, cam_intr, cam_pose, vol_bnds=None):
    voxel_size = 0.004
    if vol_bnds is None:
        vol_bnds = np.array([[0.244, 0.756],
                             [-0.256, 0.256],
                             [0.0, 0.192]])
    tsdf_vol = TSDFVolume(vol_bnds, voxel_size=voxel_size, use_gpu=True)
    tsdf_vol.integrate(color_image, depth_image, cam_intr, cam_pose, obs_weight=1.)
    volume = np.asarray(tsdf_vol.get_volume()[0])
    volume = np.transpose(volume, [1, 0, 2])
    return volume


if __name__ == '__main__':
    main()