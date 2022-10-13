import numpy as np
import os.path as osp
import math
import tqdm
import utils
from random import randint
from torch.utils.data import Dataset
from pathlib import Path
from fusion import TSDFVolume
import h5py
from utils import imretype, draw_arrow
#from fusion import TSDFVolume
import cv2
import tqdm
import json
from os import listdir
from os.path import join, isdir, isfile, splitext
import torch
import imageio
import argparse
from random import randint
from skimage import measure
from matplotlib import pyplot as plt


class ShapestacksDataset(Dataset):
    def __init__(self,
                 # base_path="/Datasets/ShapeStacks_depth/recordings",
                 # base_path="/Datasets/ShapeStacks_depth/recordings_all_cam_tiff",
                 base_path="/Datasets/ShapeStacks_depth/recordings_8fps_112x112",
                 # base_path="/Datasets/ShapeStacks_depth/recordings_large/",
                 #                 base_path="/home/karls/does_vision_matter/shapestacks_dataset/shapestacks/recordings_all_cam_tiff",
                 path_filter="easy-h=3",
                 frame_rate=1,
                 num_objects=1,
                 dummy_dataset=False,
                 overfit_index=-1,
                 context_length=0,
                 use_velocity_action=False,
                 split_size=1,
                 part_idx=0):

        # is_train=True

        allowed_cam_names = []
        self.overfit_index = overfit_index
        self.base_path = base_path
        self.num_objects = num_objects
        self.frame_rate = frame_rate
        self.direction_num = 8

        folder_names = sorted([f for f in listdir(base_path) if isdir(join(base_path, f))], reverse=True)
        self.file_names = []

        for i, d in enumerate(tqdm.tqdm(folder_names)):

            if path_filter not in d:
                continue
            # files = [f for f in listdir(join(base_path, d)) if isfile(join(base_path, d, f))]
            files = listdir(join(base_path, d))
            im_sequences = {'rgb': {}, 'depth': {}, 'iseg': {}, 'pose': {}, 'dsr': {}}
            for f in files:
                if f == 'iseg_log.txt':
                    continue

                if f == 'shapes.json':
                    continue
                if splitext(f)[1] in ['.txt', '.swp']:
                    continue
                cam_index = f.find('cam')
                cam_name = f[cam_index:f[cam_index:].find('-') + cam_index]
                if cam_name not in allowed_cam_names and len(allowed_cam_names) > 0:
                    continue
                im_type = f[:f.find('-')]
                try:
                    im_index = int(f[f.rfind('-') + 1:f.rfind('.')])
                except:
                    import pdb
                    pdb.set_trace()
                    a = 5
                if im_type in ['depth', 'iseg', 'pose']:
                    im_set_name = cam_name
                else:
                    im_set_name = f[f.find('-') + 1:f.rfind('-')]

                if im_set_name not in im_sequences[im_type]:
                    im_sequences[im_type][im_set_name] = {'cam_name': cam_name, 'ims': {}}
                im_sequences[im_type][im_set_name]['ims'][im_index] = join(self.base_path, d, f)

            for key, image_set in im_sequences['rgb'].items():
                rgb_ims = image_set['ims']
                if image_set['cam_name'] not in im_sequences['depth']:
                    continue

                #import pdb
                #pdb.set_trace()
                depth_ims = im_sequences['depth'][image_set['cam_name']]['ims']
                iseg_ims = im_sequences['iseg'][image_set['cam_name']]['ims']
                poses = im_sequences['pose'][image_set['cam_name']]['ims']

                if len(rgb_ims) != len(depth_ims):
                    continue
                self.file_names.append({'rgb': rgb_ims, 'depth': depth_ims, 'iseg': iseg_ims, 'pose': poses})

        el_per_part = split_size  # len(folder_names) // split_size
        self.file_names = self.file_names[part_idx * el_per_part:(part_idx + 1) * el_per_part]

        # Constants taken from the simulator
        height = 112
        width = 112
        # self.fovy = 90
        self.fovy = 45
        f = 0.5 * height / math.tan(self.fovy * math.pi / 360)

        self.camera_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))

        # More constants from the simulator
        self.extent = 4.0

        znear = 0.10000000149011612
        zfar = 50.0
        self.depth_near = znear  # * self.extent
        self.depth_far = zfar  # * self.extent

        self.voxel_size = 10/128
        #self.view_bounds = np.array([[-5.0, 5.0],
        #                            [-5.0, 5.0],
        #                            [0.0, 5.0]], dtype=np.float32)

        self.view_bounds = np.array([[-5.0, 5.0],
                                    [-5.0, 5.0],
                                    [0.0, 5.0]], dtype=np.float32)

        #self.volume_size = np.array([128, 128, 64])
        self.volume_size = ((self.view_bounds[:, 1] - self.view_bounds[:, 0]) / self.voxel_size).astype(int)

        #array([[-2.1344433, 2.7787285],
        #       [-1.4454694, 9.300999],
        #       [0.282004, 3.1099832]], dtype=float32)


    #        focal_scaling = (1./np.tan(np.deg2rad(fovy)/2)) * height / 2.0
    #        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    #        print("focal", focal)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # idx = 8
        if self.overfit_index >= 0:
            idx = self.overfit_index
        depth_ims = self.file_names[idx]['depth']
        rgb_ims = self.file_names[idx]['rgb']
        iseg_ims = self.file_names[idx]['iseg']
        poses = self.file_names[idx]['pose']
        self.sequence_length = len(depth_ims.keys())

        # Get the information about the gt shape
        shapes_info_fn = Path(self.file_names[idx]['depth'][0]).parent / 'shapes.json'
        with open(shapes_info_fn, 'r') as fs:
            shapes_info = json.load(fs)

            res = 128
            qp = torch.stack(torch.meshgrid(
                torch.linspace(-1.0, 1.0, res).cuda(),
                torch.linspace(-1.0, 1.0, res).cuda(),
                torch.linspace(-1.0, 1.0, res).cuda(),
            ), dim=-1).reshape(-1, 3)

            for i in range(len(shapes_info)):

                shape_sdf = self.get_shape_sdf_cuda(qp, shapes_info[i])

                gt_verts, gt_faces, _, _ = measure.marching_cubes_lewiner(shape_sdf.reshape(res, res, res).cpu().numpy(), 0)
                gt_verts = 2 * gt_verts / (res - 1) - 1
                ShapestacksDataset.write_obj(f'/home/diegopc/tmp/a1234/gt_mesh_shapestacks_idx_{idx}_obj_{i}.obj', gt_verts, gt_faces + 1)

                shapes_info[i]['verts'] = gt_verts
                shapes_info[i]['faces'] = gt_faces
                shapes_info[i]['coords'] = qp[shape_sdf <= 0.0].cpu().numpy()

        gt_T = np.zeros((self.sequence_length, self.num_objects, 4, 4), dtype=np.float32)
        world_gt_T = np.zeros((self.sequence_length, self.num_objects, 4, 4), dtype=np.float32)
        ims = []
        seg_masks = []
        start_index = 0
        sample = {}
        for j in range(self.sequence_length):
            i = start_index + self.frame_rate * j
            im = cv2.imread(rgb_ims[i])

            # Reads the depth image with values in [-1.0, 1.0]
            depth_im = 2 * imageio.imread(depth_ims[i]) - 1.0
            depth_im = depth_im.astype(np.float32)

            try:
                with open(poses[i], 'r') as f:
                    pose_data = json.load(f)
            except Exception as ex:
                print("poses[i]", poses[i])
                raise ex

            camera_pose = torch.zeros(4, 4)
            camera_pose[3, 3] = 1.0
            camera_pose[:3, :3] = ShapestacksDataset.quaternion_to_matrix(torch.tensor(pose_data['camera_quat']))
            # camera_pose[:3, :3] = ShapestacksDataset.quaternion_to_matrix(torch.tensor([pose_data['camera_quat'][3], pose_data['camera_quat'][0], pose_data['camera_quat'][1], pose_data['camera_quat'][2]]))
            camera_pose[:3, 3] = torch.tensor(pose_data['camera_pos'])
            inverse_camera_pose = torch.inverse(camera_pose)

            # Transforms from mujoco's camera convention to ours
            mujoco_transform = torch.zeros(4, 4)
            mujoco_transform[3, 3] = 1.
            mujoco_transform[0, 0] = 1.
            mujoco_transform[1, 2] = -1.
            mujoco_transform[2, 1] = 1.

            for obj_index in range(len(pose_data['object_pose']) // 7):
                object_pose = torch.zeros(4, 4)
                # object_pose[:3, :3] = torch.eye(3)
                object_pose[3, 3] = 1.0
                #                object_pose[1, 3] += 1.0
                #                object_pose = torch.tensor([[-0.23368588,  0.20228636,  0.9510369 , -0.26653785],
                #  [-0.31944436,  0.9078504,  -0.27159345,  1.8415115 ],
                #  [-0.91833884, -0.36727092, -0.14753257,  0.34274215],
                #  [ 0.        ,  0.        ,  0.        ,  1.        ]])
                object_pose[:3, :3] = ShapestacksDataset.quaternion_to_matrix(
                    torch.tensor(pose_data['object_pose'][obj_index * 7 + 3:obj_index * 7 + 7]))
                object_pose[:3, 3] = torch.tensor(pose_data['object_pose'][obj_index * 7:obj_index * 7 + 3])
                # print("object_pose", object_pose)
                # print("inverse cam", inverse_camera_pose)
                # print("object with inv", inverse_camera_pose @ object_pose)

                object_pose_in_camera = mujoco_transform @ inverse_camera_pose @ object_pose
                # print("object_pose_in_camera", object_pose_in_camera)
                # object_pose_in_camera[:3, :3] = torch.eye(3)
                object_pose_in_camera[:3, 3] /= self.extent

                gt_T[j, obj_index] = mujoco_transform @ object_pose_in_camera
                world_gt_T[j, obj_index] = object_pose

                # gt_T[j, obj_index] = object_pose

            # Converts to real depth (positive depth)
            depth_im = -ShapestacksDataset.get_denormalized_depth(depth_im, zfar=self.depth_far, znear=self.depth_near)
            depth_im = np.clip(depth_im, a_min=0.0, a_max=self.extent)

            # Reads the segmentation masks
            mask = imageio.imread(iseg_ims[i]) > 0.0
            foreground = (~(mask[..., 2] & mask[..., 1] & mask[..., 0]))  # .astype(np.float32)
            mask = np.stack([mask[..., 0] & foreground, mask[..., 1] & foreground, mask[..., 2] & foreground])

            if self.num_objects == 1:
                mask = np.any(mask, axis=0, keepdims=True)
            elif self.num_objects <= 3:
                # uses only the first self.num_objects
                mask = mask[:self.num_objects, ...]
            else:
                raise ValueError(f'Currently the ground truth segmentation only support 3 objects per scene.')

            # mask = np.ones_like(depth_im).astype(np.float32)
            seg_masks.append(mask)
            # depth_im = depth_im * mask + (1.0 - mask) * self.extent

            # with open('/home/diegopc/tmp/cube_rendering.npy', 'rb') as f:
            #    #depth_im = torch.load(f'/home/diegopc/tmp/cube_rendering.pt').detach().cpu().numpy()[0, ...]
            #    depth_im = np.load(f)[0, ...]
            #    mask = (~(depth_im == self.extent))
            #    seg_masks.append(mask)

            # Convert to meters
            # https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L858
            # depth_im = (self.depth_near / (1 - depth_im * (1 - self.depth_near / self.depth_far))).astype(np.float32)
            #im = np.concatenate((im, depth_im.reshape(1, depth_im.shape[0], depth_im.shape[1])))

            #pose_data['camera_pos']
            #pose_data['camera_quat']
            #pose_data['object_pos']
            #gt_T


            tsdf = ShapestacksDataset.get_volume(
                color_image=im,
                depth_image=depth_im,
                cam_intr=self.camera_matrix,
                #cam_pose= camera_pose.numpy() @ cam_align,
                #cam_pose= mujoco_transform.numpy() @ camera_pose.numpy(),
                cam_pose=np.eye(4),
                #cam_pose=(inverse_camera_pose @ mujoco_transform.T).numpy(),
                #cam_pose=inverse_camera_pose.numpy() @ mujoco_transform.numpy().T,
                vol_bnds=self.view_bounds,
                voxel_size=self.voxel_size,
            )

            #plt.imshow(depth_im)
            #plt.colorbar()
            #plt.savefig('/home/diegopc/tmp/tsdf_shapestacks.png')

            pred_verts, pred_faces, _, _ = measure.marching_cubes_lewiner(tsdf, 0)
            pred_verts = pred_verts * self.voxel_size + self.view_bounds[:, 0]

            ShapestacksDataset.write_obj(f'/home/diegopc/tmp/a1234/tsdf_shapestacks_idx_{idx}_step_id_{j}.obj', pred_verts, pred_faces + 1)

            '''import pdb
            pdb.set_trace()

            f.create_dataset('dsr_data', data=tsdf, compression="gzip", compression_opts=4)'''

            if j > 0:
                #self._get_scene_flow_3d(old_pos, old_ori, new_pos, new_ori, num_objects)
                mask_3d, flow_3d = self._get_scene_flow_3d(
                    gt_T[j - 1, :, :3, 3], gt_T[j - 1, :, :3, :3], gt_T[j, :, :3, 3], gt_T[j, :, :3, :3],
                    #world_gt_T[j - 1, :, :3, 3], world_gt_T[j - 1, :, :3, :3], world_gt_T[j, :, :3, 3], world_gt_T[j, :, :3, :3],
                    self.num_objects,
                    shapes_info,
                )
                pred_verts, pred_faces, _, _ = measure.marching_cubes_lewiner(-(mask_3d > 0).astype(int) + 0.5, 0)
                pred_verts = pred_verts * self.voxel_size + self.view_bounds[:, 0]
                ShapestacksDataset.write_obj(f'/home/diegopc/tmp/a1234/mask_shapestacks_idx_{idx}_step_id_{j}.obj', pred_verts, pred_faces + 1)

                #import pdb
                #pdb.set_trace()

            else:
                mask_3d = np.zeros(self.volume_size).astype(np.int64)
                flow_3d = np.zeros([3] + self.volume_size.tolist()).astype(np.float32)


            sample['world_T'] = world_gt_T
            sample['gt_T'] = world_gt_T
            #sample['T'][..., 3, 3] = 1.0
            #sample['T'][..., :3, :3] = gt_T[..., :3, :3].permute(0, 1, 3, 2)
            #sample['T'][..., :3, 3:] = - sample['T'][..., :3, :3] @ gt_T[..., :3, 3:]



            im = cv2.resize(im, (240, 240))
            depth_im = cv2.resize(depth_im, (240, 240), interpolation=cv2.INTER_NEAREST)
            ims.append(im)

            #
            #sample['%d-action' % j] = np.zeros(shape=[self.direction_num, self.volume_size[0], self.volume_size[1]], dtype=np.float32)
            #sample['%d-color_image' % j] = ShapestacksDataset.pad_image(im, size=(240, 320, 3), pad_value=0)
            #sample['%d-depth_image' % j] = ShapestacksDataset.pad_image(depth_im, size=(240, 320), pad_value=self.extent)
            #sample['%d-color_heightmap' % j] = (255 * np.ones_like(sample['%d-color_image' % j])).astype('uint8')

            sample['%d-tsdf' % j] = np.zeros(self.volume_size).astype(np.float32)
            # sample['%d-tsdf' % count_seq] = np.zeros((128, 128, 128), dtype=np.float32)
            sample['%d-mask_3d' % j] = np.zeros(self.volume_size).astype(np.int64)
            sample['%d-scene_flow_3d' % j] = np.zeros([3] + self.volume_size.tolist()).astype(np.float32)

            #import pdb
            #pdb.set_trace()

            dsr_data_fn = Path(depth_ims[i]).name.replace('depth', 'dsr').replace('.tiff', '.hdf5')
            f5_file = h5py.File(Path(depth_ims[i]).parent / dsr_data_fn, 'w')
            f5_file.create_dataset(f'tsdf', data=tsdf, compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'mask_3d', data=mask_3d, compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'scene_flow_3d', data=flow_3d, compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'view_bounds', data=self.view_bounds, compression="gzip", compression_opts=4)
            for u in range(len(shapes_info)):
                f5_file.create_dataset(f'{u}-verts', data=shapes_info[u]['verts'], compression="gzip", compression_opts=4)
                f5_file.create_dataset(f'{u}-faces', data=shapes_info[u]['faces'], compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'world_T', data=world_gt_T[j], compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'gt_T', data=gt_T[j], compression="gzip", compression_opts=4)
            f5_file.close()

            #print(f'created: {Path(depth_ims[i]).parent / dsr_data_fn}')

        return sample

    def _get_scene_flow_3d(self, old_pos, old_ori, new_pos, new_ori, num_objects, shapes_info):

        vol_bnds = self.view_bounds
        scene_flow = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds] + [3])
        mask = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds], dtype=np.int)

        cur_cnt = 0
        for s_info, old_p, old_o, new_p, new_o in zip(shapes_info, old_pos, old_ori, new_pos, new_ori):
            #import pdb
            #pdb.set_trace()

            new_coord = self._get_coord(s_info, new_p, new_o, vol_bnds, self.voxel_size)
            old_coord = self._get_coord(s_info, old_p, old_o, vol_bnds, self.voxel_size)

            motion = new_coord - old_coord

            valid_idx = np.logical_and(
                np.logical_and(old_coord[:, 1] >= 0, old_coord[:, 1] < mask.shape[1]),
                np.logical_and(
                    np.logical_and(old_coord[:, 0] >= 0, old_coord[:, 0] < mask.shape[0]),
                    np.logical_and(old_coord[:, 2] >= 0, old_coord[:, 2] < mask.shape[2])
                )
            )
            x = old_coord[valid_idx, 1]
            y = old_coord[valid_idx, 0]
            z = old_coord[valid_idx, 2]
            motion = motion[valid_idx]
            motion = np.stack([motion[:, 1], motion[:, 0], motion[:, 2]], axis=1)

            scene_flow[x, y, z] = motion

            # mask
            cur_cnt += 1
            mask[x, y, z] = cur_cnt

        return mask, scene_flow

    def maximum(self, tensor, value):

        if isinstance(value, torch.Tensor):
            return torch.where(tensor > value, tensor, value)
        else:
            return torch.where(tensor > value, tensor, value * torch.ones_like(tensor))

    def minimum(self, tensor, value):
        if isinstance(value, torch.Tensor):
            return torch.where(tensor < value, tensor, value)
        else:
            return torch.where(tensor < value, tensor, value * torch.ones_like(tensor))

    def get_shape_sdf_cuda(self, query_points, shape_info):

        if shape_info['type'] == 'box':
            q = query_points.abs() - torch.tensor(shape_info['size']).cuda() / self.extent
            sdf = self.maximum(q, 0.0).norm(dim=-1) + self.minimum(self.maximum(q[:, 0], self.maximum(q[:, 1], q[:, 2])), 0.0)
            return sdf
        elif shape_info['type'] == 'sphere':
            return query_points.norm(dim=-1) - shape_info['size'][0] / self.extent
        elif shape_info['type'] == 'cylinder':
            d = torch.stack([query_points[:, [0, 1]].norm(dim=-1), query_points[:, 2]]).abs().T
            d = d - torch.tensor(shape_info['size']).cuda() / self.extent
            return self.minimum(self.maximum(d[:, 0], d[:, 1]), 0.0) + self.maximum(d, 0.0).norm(dim=-1)
        else:
            raise ValueError(f'Invalid shape type: {shape_info["type"]}.')

    def _get_coord(self, shape_info, position, orientation, vol_bnds=None, voxel_size=None):

        '''qp = np.stack(np.meshgrid(
            np.linspace(self.view_bounds[0, 0], self.view_bounds[0, 1], self.volume_size[0]),
            np.linspace(self.view_bounds[1, 0], self.view_bounds[1, 1], self.volume_size[1]),
            np.linspace(self.view_bounds[2, 0], self.view_bounds[2, 1], self.volume_size[2]),
        ), axis=-1).reshape(-1, 3)'''

        #import pdb
        #pdb.set_trace()

        '''res = 128
        qp = torch.stack(torch.meshgrid(
            torch.linspace(-1.0, 1.0, res).cuda(),
            torch.linspace(-1.0, 1.0, res).cuda(),
            torch.linspace(-1.0, 1.0, res).cuda(),
        ), dim=-1).reshape(-1, 3)

        shape_sdf = self.get_shape_sdf_cuda(qp, shape_info)

        coord = qp[shape_sdf <= 0.0].cpu().numpy()'''

        coord = shape_info['coords']


        #pt = np.random.rand(10000, 3) - 0.5
        #pt /= np.linalg.norm(pt, axis=-1, keepdims=True)
        #pt *= 0.15

        #coord = pt

        # if vol_bnds is not None, return coord in voxel, else, return world coord
        #coord = self.voxel_coord[obj_id]
        mat = orientation
        coord = (mat @ (coord.T)).T + np.asarray(position)
        if vol_bnds is not None:
            coord = np.round((coord - vol_bnds[:, 0]) / voxel_size).astype(np.int)

        return coord

    @staticmethod
    def write_obj(filepath, verts, faces):
        with open(filepath, 'w') as f:
            f.write("# OBJ file\n")
            for v in verts:
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for fc in faces:
                f.write(f"f {fc[0]:d} {fc[1]:d} {fc[2]:d}\n")

    @staticmethod
    def get_volume(color_image, depth_image, cam_intr, cam_pose, vol_bnds=None, voxel_size=0.01):
        if vol_bnds is None:
            vol_bnds = np.array([[0.244, 0.756],
                                 [-0.256, 0.256],
                                 [0.0, 0.192]])
        tsdf_vol = TSDFVolume(vol_bnds, voxel_size=voxel_size, use_gpu=True)
        tsdf_vol.integrate(color_image, depth_image, cam_intr, cam_pose, obs_weight=1.)
        volume = np.asarray(tsdf_vol.get_volume()[0])
        volume = np.transpose(volume, [1, 0, 2])
        return volume

    @staticmethod
    def pad_image(img, size, pad_value=0):

        H_pad = (size[0] - img.shape[0]) // 2
        W_pad = (size[1] - img.shape[1]) // 2
        padded_img = np.ones(size, dtype=img.dtype) * pad_value
        padded_img[H_pad:H_pad + img.shape[0], W_pad:W_pad + img.shape[1]] = img

        return padded_img

    @staticmethod
    def get_denormalized_depth(z_n, zfar, znear):
        '''
            Follows: http://www.songho.ca/opengl/gl_projectionmatrix.html
            Assumes the input depth is normalized in the range [-1, 1]. The values of zfar and znear are positive
        '''
        A = - (zfar + znear) / (zfar - znear)
        B = - 2 * zfar * znear / (zfar - znear)
        z_e = -B / (A + z_n)
        return z_e

    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--object_num', type=int, default=5, help='number of objects')
    parser.add_argument('--split_size', type=int, default=20, help='size of the part')
    parser.add_argument('--part_idx', type=int, default=0, help='Part index')

    args = parser.parse_args()

    ds = ShapestacksDataset(base_path=args.data_path, num_objects=args.object_num, split_size=args.split_size, part_idx=args.part_idx)

    T = []
    for i in tqdm.tqdm(range(ds.__len__())):

        #import pdb
        #pdb.set_trace()
        print(f'Processed sample {i}.')
        sample = ds.__getitem__(i)

        #T.append(sample['T'])
    #import pdb
    #pdb.set_trace()

