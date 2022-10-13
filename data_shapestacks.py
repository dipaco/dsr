import numpy as np
import os.path as osp
import math
import tqdm
import utils
from random import randint
from torch.utils.data import Dataset
from pathlib import Path
#from fusion import TSDFVolume
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
from random import randint
from skimage import measure


class ShapestacksDataset(Dataset):
    def __init__(self,
                 # base_path="/Datasets/ShapeStacks_depth/recordings",
                 # base_path="/Datasets/ShapeStacks_depth/recordings_all_cam_tiff",
                 base_path="/Datasets/ShapeStacks_depth/recordings_8fps_112x112",
                 # base_path="/Datasets/ShapeStacks_depth/recordings_large/",
                 #                 base_path="/home/karls/does_vision_matter/shapestacks_dataset/shapestacks/recordings_all_cam_tiff",
                 path_filter="easy-h=3",
                 split='train',
                 frame_rate=1,
                 num_objects=1,
                 dummy_dataset=False,
                 overfit_index=-1,
                 sequence_length=1,
                 context_length=0,
                 use_velocity_action=False):

        # is_train=True

        allowed_cam_names = []
        self.overfit_index = overfit_index
        self.base_path = base_path
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.is_train = split == 'train'
        self.num_objects = num_objects
        self.frame_rate = frame_rate
        self.min_frames = frame_rate * (self.sequence_length + self.context_length)
        self.direction_num = 8
        self.volume_size = np.array([128, 128, 64])

        folder_names = [f for f in listdir(base_path) if isdir(join(base_path, f))]


        #import pdb
        #pdb.set_trace()

        self.file_names = []
        for i, d in enumerate(tqdm.tqdm(folder_names)):
            if (i % 10 == 0 and self.is_train) or (i % 10 != 0 and not self.is_train):
                continue

            if dummy_dataset and len(self.file_names) > 20:
                break

            if path_filter not in d:
                continue
            # files = [f for f in listdir(join(base_path, d)) if isfile(join(base_path, d, f))]
            files = listdir(join(base_path, d))
            im_sequences = {'rgb': {}, 'depth': {}, 'iseg': {}, 'pose': {}, 'dsr': {}}
            for f in files:
                if f == 'shapes.json':
                    continue
                if splitext(f)[1] == '.txt':
                    continue
                cam_index = f.find('cam')
                cam_name = f[cam_index:f[cam_index:].find('-') + cam_index]
                if cam_name not in allowed_cam_names and len(allowed_cam_names) > 0:
                    continue
                im_type = f[:f.find('-')]
                im_index = int(f[f.rfind('-') + 1:f.rfind('.')])
                if im_type in ['depth', 'iseg', 'pose', 'dsr']:
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
                dsr = im_sequences['dsr'][image_set['cam_name']]['ims']

                if len(rgb_ims) < self.min_frames or len(rgb_ims) != len(depth_ims):
                    continue
                self.file_names.append({'rgb': rgb_ims, 'depth': depth_ims, 'iseg': iseg_ims, 'pose': poses, 'dsr': dsr})

        #import pdb
        #pdb.set_trace()

        # Constants taken from the simulator
        height = 224
        width = 224
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

        self.__getitem__(0)

    #        focal_scaling = (1./np.tan(np.deg2rad(fovy)/2)) * height / 2.0
    #        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    #        print("focal", focal)

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def get_shape_sdf_cuda(query_points, shape_info):
        extent = 4.0

        if shape_info['type'] == 'box':
            q = query_points.abs() - torch.tensor(shape_info['size']) / extent
            sdf = ShapestacksDataset.maximum(q, 0.0).norm(dim=-1) + ShapestacksDataset.minimum(ShapestacksDataset.maximum(q[:, 0], ShapestacksDataset.maximum(q[:, 1], q[:, 2])), 0.0)
            return sdf
        elif shape_info['type'] == 'sphere':
            return query_points.norm(dim=-1) - shape_info['size'][0] / extent
        elif shape_info['type'] == 'cylinder':
            d = torch.stack([query_points[:, [0, 1]].norm(dim=-1), query_points[:, 2]]).abs().T
            d = d - torch.tensor(shape_info['size']) / extent
            return ShapestacksDataset.minimum(ShapestacksDataset.maximum(d[:, 0], d[:, 1]), 0.0) + ShapestacksDataset.maximum(d, 0.0).norm(dim=-1)
        else:
            raise ValueError(f'Invalid shape type: {shape_info["type"]}.')

    @staticmethod
    def maximum(tensor, value):

        if isinstance(value, torch.Tensor):
            return torch.where(tensor > value, tensor, value)
        else:
            return torch.where(tensor > value, tensor, value * torch.ones_like(tensor))

    @staticmethod
    def minimum(tensor, value):
        if isinstance(value, torch.Tensor):
            return torch.where(tensor < value, tensor, value)
        else:
            return torch.where(tensor < value, tensor, value * torch.ones_like(tensor))

    def __getitem__(self, idx):
        # idx = 8
        if self.overfit_index >= 0:
            idx = self.overfit_index
        depth_ims = self.file_names[idx]['depth']
        rgb_ims = self.file_names[idx]['rgb']
        iseg_ims = self.file_names[idx]['iseg']
        poses = self.file_names[idx]['pose']
        dsr_data = self.file_names[idx]['dsr']

        view_bounds = np.array([[-5.0, 5.0],
                                [-5.0, 5.0],
                                [0.0, 5.0]], dtype=np.float32)

        # Get the information about the gt shape
        shapes_info_fn = Path(self.file_names[idx]['depth'][0]).parent / 'shapes.json'
        with open(shapes_info_fn, 'r') as fs:
            shapes_info = json.load(fs)

            res = 64
            qp = torch.stack(torch.meshgrid(
                torch.linspace(-1.0, 1.0, res),
                torch.linspace(-1.0, 1.0, res),
                torch.linspace(-1.0, 1.0, res),
            ), dim=-1).reshape(-1, 3)

            for i in range(len(shapes_info)):
                shape_sdf = ShapestacksDataset.get_shape_sdf_cuda(qp, shapes_info[i])

                gt_verts, gt_faces, _, _ = measure.marching_cubes_lewiner(
                    shape_sdf.reshape(res, res, res).cpu().numpy(), 0)
                gt_verts = 2 * gt_verts / (res - 1) - 1
                #ShapestacksDataset.write_obj(f'/home/diegopc/tmp/a1234/gt_mesh_shapestacks_idx_{idx}_obj_{i}.obj', gt_verts, gt_faces + 1)

                shapes_info[i]['verts'] = gt_verts
                shapes_info[i]['faces'] = gt_faces


        gt_T = np.zeros((self.sequence_length + self.context_length, self.num_objects, 4, 4), dtype=np.float32)
        ims = []
        seg_masks = []

        start_index = randint(1, len(rgb_ims) - self.min_frames - 1)
        # start_index = 0
        sample = {}
        for j in range(self.sequence_length + self.context_length):
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
                gt_T[j, obj_index] = object_pose_in_camera

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
            im = cv2.resize(im, (240, 240))
            depth_im = cv2.resize(depth_im, (240, 240), interpolation=cv2.INTER_NEAREST)
            ims.append(im)

            #import pdb
            #pdb.set_trace()

            #dsr_data_fn = Path(depth_ims[i]).name.replace('depth', 'dsr').replace('.tiff', '.hdf5')
            #f5_file = h5py.File(Path(depth_ims[i]).parent / dsr_data_fn, 'r')
            f5_file = h5py.File(dsr_data[i], 'r')

            # Dsr data of previous step
            #dsr_data_prev_fn = Path(depth_ims[i - 1]).name.replace('depth', 'dsr').replace('.tiff', '.hdf5')
            #f5_file_prev = h5py.File(Path(depth_ims[i - 1]).parent / dsr_data_prev_fn, 'r')
            f5_file_prev = h5py.File(dsr_data[i - 1], 'r')

            '''f5_file.create_dataset(f'tsdf', data=tsdf, compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'mask_3d', data=mask_3d, compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'scene_flow_3d', data=flow_3d, compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'view_bounds', data=self.view_bounds, compression="gzip", compression_opts=4)
            for u in range(len(shapes_info)):
                f5_file.create_dataset(f'{u}-verts', data=shapes_info[u]['verts'], compression="gzip", compression_opts=4)
                f5_file.create_dataset(f'{u}-faces', data=shapes_info[u]['faces'], compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'world_T', data=world_gt_T[j], compression="gzip", compression_opts=4)
            f5_file.create_dataset(f'gt_T', data=gt_T[j], compression="gzip", compression_opts=4)'''


            #
            sample['%d-action' % j] = np.zeros(shape=[self.direction_num, self.volume_size[0], self.volume_size[1]], dtype=np.float32)
            sample['%d-color_image' % j] = ShapestacksDataset.pad_image(im, size=(240, 320, 3), pad_value=0)
            sample['%d-depth_image' % j] = ShapestacksDataset.pad_image(depth_im, size=(240, 320), pad_value=self.extent)
            sample['%d-color_heightmap' % j] = (255 * np.ones_like(sample['%d-color_image' % j])).astype('uint8')

            sample['%d-tsdf' % j] = np.asarray(f5_file_prev['tsdf']).astype(np.float32)
            sample['%d-mask_3d' % j] = np.asarray(f5_file['mask_3d']).astype(np.int64)
            sample['%d-scene_flow_3d' % j] = np.asarray(f5_file['scene_flow_3d']).astype(np.float32).transpose([3, 0, 1, 2])

            #sample['faces'] = [f5_file['%d-faces' % u] for u in range(self.num_objects)]

            #import pdb
            #pdb.set_trace()
            sample['faces'] = [shapes_info[u]['faces'] for u in range(self.num_objects)]

            #gt_T[j, :, :3, 3], gt_T[j, :, :3, :3]
            #coord = (gt_T[j, u, :3, :3] @ (shapes_info[u]['verts'].T)).T + gt_T[j, u, :3, 3]


            sample['%d-verts' % j] = [
                (gt_T[j, u, :3, :3] @ (shapes_info[u]['verts'].T)).T + gt_T[j, u, :3, 3] for u in range(self.num_objects)
            ]

            '''sample['%d-tsdf' % j] = np.zeros(self.volume_size).astype(np.float32)
            # sample['%d-tsdf' % count_seq] = np.zeros((128, 128, 128), dtype=np.float32)
            sample['%d-mask_3d' % j] = np.zeros(self.volume_size).astype(np.int64)
            sample['%d-scene_flow_3d' % j] = np.zeros([3] + self.volume_size.tolist()).astype(np.float32)'''


        ims = np.stack(ims)
        seg_masks = np.stack(seg_masks)

        gt_T = gt_T[:, ::-1].copy()
        context_gt_T = gt_T[:self.context_length]
        target_gt_T = gt_T[self.context_length:]




        return sample


        sample = {'target_frames': ims[self.context_length:],
                  'actions': actions[self.context_length - 1:],
                  'context_frames': ims[:self.context_length],
                  'context_frames_masks': seg_masks[:self.context_length],
                  'target_frames_masks': seg_masks[self.context_length:],
                  'context_actions': actions[:self.context_length],
                  'camera_matrix': self.camera_matrix,
                  'depth_near': self.depth_near,
                  'depth_far': self.depth_far,
                  'max_depth': self.extent,
                  'fov': self.fovy,
                  'context_gt_T': context_gt_T,
                  'target_gt_T': target_gt_T,
                  }
        return sample

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