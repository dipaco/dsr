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


class Data(Dataset):
    def __init__(self, data_path, split, seq_len):
        self.data_path = data_path
        self.tot_seq_len = 10
        self.seq_len = seq_len
        self.volume_size = [128, 128, 48]
        self.direction_num = 8
        self.voxel_size = 0.004
        self.idx_list = open(osp.join(self.data_path, '%s.txt' % split)).read().splitlines()
        self.returns = ['action', 'color_heightmap', 'color_image', 'tsdf', 'mask_3d', 'scene_flow_3d']
        self.data_per_seq = self.tot_seq_len // self.seq_len

    def __getitem__(self, index):
        data_dict = {}
        idx_seq = index // self.data_per_seq
        idx_step = index % self.data_per_seq * self.seq_len
        for step_id in range(self.seq_len):
            f = h5py.File(osp.join(self.data_path, "%s_%d.hdf5" % (self.idx_list[idx_seq], idx_step + step_id)), "r")

            # action
            action = f['action']
            data_dict['%d-action' % step_id] = self.get_action(action)

            # color_image, [W, H, 3]
            if 'color_image' in self.returns:
                data_dict['%d-color_image' % step_id] = np.asarray(f['color_image_small'], dtype=np.uint8)

            # color_heightmap, [128, 128, 3]
            if 'color_heightmap' in self.returns:
                # draw arrow for visualization
                color_heightmap = draw_arrow(
                    np.asarray(f['color_heightmap'], dtype=np.uint8),
                    (int(action[2]), int(action[1]), int(action[0]))
                )
                data_dict['%d-color_heightmap' % step_id] = color_heightmap

            # tsdf, [S1, S2, S3]
            if 'tsdf' in self.returns:
                data_dict['%d-tsdf' % step_id] = np.asarray(f['tsdf'], dtype=np.float32)

            # mask_3d, [S1, S2, S3]
            if 'mask_3d' in self.returns:
                data_dict['%d-mask_3d' % step_id] = np.asarray(f['mask_3d'], dtype=np.int)

            # scene_flow_3d, [3, S1, S2, S3]
            if 'scene_flow_3d' in self.returns:
                scene_flow_3d = np.asarray(f['scene_flow_3d'], dtype=np.float32).transpose([3, 0, 1, 2])
                data_dict['%d-scene_flow_3d' % step_id] = scene_flow_3d

        return data_dict

    def __len__(self):
        return len(self.idx_list) * self.data_per_seq

    def get_action(self, action):
        direction, r, c = int(action[0]), int(action[1]), int(action[2])
        if direction < 0:
            direction += self.direction_num
        action_map = np.zeros(shape=[self.direction_num, self.volume_size[0], self.volume_size[1]], dtype=np.float32)
        action_map[direction, r, c] = 1

        return action_map


class ParabolicShotData(Dataset):
    def __init__(self, data_path, split, seq_len, context_length=1, overfit_index=-1, use_velocity_action=False, mask_out_bg=False):

        self.sequence_length = seq_len
        self.context_length = context_length
        self.use_velocity_action = use_velocity_action
        self.split = split
        self.mask_out_bg = mask_out_bg

        self.min_frames = self.sequence_length
        self.overfit_index = overfit_index
        if self.overfit_index > -1:
            self.split = 'train'

        self.image_size = 224
        self.fovy = 45  # 90#45
        f = 0.5 * self.image_size / math.tan(self.fovy * math.pi / 360)
        self.camera_matrix = np.array(((f, 0, self.image_size / 2), (0, f, self.image_size / 2), (0, 0, 1)))

        base = Path(data_path)
        seq_paths_train = []
        seq_paths_test = []
        seq_paths_validation = []
        all_paths = list(sorted(base.iterdir()))
        for i, el in enumerate(tqdm.tqdm(all_paths)):

            if not el.is_dir():
                continue

            if i % 10 in range(0, 7):
                seq_paths_train.append(el)
            elif i % 10 in range(7, 9):
                seq_paths_validation.append(el)
            else:
                seq_paths_test.append(el)

        if self.split == 'train':
            self.seq_paths = seq_paths_train
        elif self.split == 'val':
            self.seq_paths = seq_paths_validation
        else:
            self.seq_paths = seq_paths_test

        # FIXME: Read this from the base folder
        self.num_objects = 1
        self.max_depth = 4.0
        self.camera_translation = 2.5
        self.volume_size = [128, 128, 128]
        self.direction_num = 8
        self.world_extents = 5
        self.view_bounds = self.world_extents / 2 * np.array([[-0.7071, 0.7071],
                                    [-0.7071, 0.7071],
                                    [-0.7071, 0.7071]])
        self.view_bounds = np.array([[-0.72, 0.72], [-0.72, 0.72], [-0.72, 0.72]])  # * self.camDistance
        self.voxel_size = (self.view_bounds[:, 1] - self.view_bounds[:, 0]).max() / self.volume_size[0]  # 0.01125  # * self.camDistance
        #self.tsdf_vol = TSDFVolume(self.view_bounds, voxel_size=self.voxel_size, use_gpu=True)


        #for i in range(1000):
        #    print(i)
        #    a = self.__getitem__(i)
        self.__getitem__(0)

        #import pdb
        #pdb.set_trace()

    def __getitem__(self, idx):

        sample = {}

        # Overfit to this example
        if self.overfit_index > -1:
            idx = self.overfit_index

        total_len = self.sequence_length# + self.context_length
        image = np.zeros((total_len, 4, self.image_size, self.image_size), dtype=np.float32)
        segmentation_masks = np.zeros((total_len, self.num_objects, self.image_size, self.image_size), dtype=np.float32)

        # Max depth
        image[:, 3, :, :] = self.max_depth

        actions = np.zeros((total_len - 1, 3))
        #actions = actions.astype(np.float32)

        # Read from disk
        current_path = self.seq_paths[idx]

        with np.load(current_path / 'data.npz', allow_pickle=True) as data:

            assert self.min_frames <= data['depth'].shape[
                0], "The requested sequence is longer than the available number frames."

            # Overfit to this example
            if self.overfit_index > -1:
                start_index = 0
            else:
                # ignores the first element in the sequence
                start_index = randint(1, data['depth'].shape[0] - self.min_frames)

            if self.split == 'test':
                start_index = 1



            #idx_seq = start_index + self.frame_rate * np.arange(self.sequence_length + self.context_length)
            idx_seq = start_index + np.arange(self.sequence_length)
            #idx_seq = np.arange(self.sequence_length)

            depth = data['depth']
            color = data['rgb']
            mask = data['mask']
            verts = data['verts']
            faces = data['faces']

            '''tsdf = data['dsr_data'].tolist()['tsdf_volume']
            camera_pose = data['dsr_data'].tolist()['camera_pose']
            camera_intrinsics = data['dsr_data'].tolist()['camera_intrinsics']
            mask_3d_volume = data['dsr_data'].tolist()['mask_3d_volume']
            flow_3d_volume = data['dsr_data'].tolist()['flow_3d_volume']
            height_map = data['dsr_data'].tolist()['height_map']
            view_bounds = data['dsr_data'].tolist()['view_bounds']
            tsdf = data['dsr_data'].tolist()['tsdf_volume']
            #depth = depth[k], mask = mask[k], rgb = rgb[k], T = T[k], dsr_data = dsr_data[k])'''

            if 'T' in data:
                gt_T = data['T'].astype(np.float32)
            else:
                gt_T = np.eye(4)[None].repeat(depth.shape[0], axis=0).astype(np.float32)

        h5_file = h5py.File(current_path / 'dsr_data.hdf5', "r")
        mask_3d_volume = np.asarray(h5_file['mask_3d_volume'], dtype=np.int) #np.asarray(mask_3d, dtype=np.int)
        flow_3d_volume = np.asarray(h5_file['flow_3d_volume'], dtype=np.float32).transpose([0, 4, 1, 2, 3]) #np.asarray(scene_flow_3d, dtype=np.float32).transpose([3, 0, 1, 2])
        height_map = np.asarray(h5_file['height_map'], dtype=np.uint8)
        tsdf = np.asarray(h5_file['tsdf_volume'], dtype=np.float32)
        cur_vel = np.asarray(h5_file['cur_vel'], dtype=np.float32)
        obj_rot = np.asarray(h5_file['obj_rot'], dtype=np.float32)
        obj_pos = np.asarray(h5_file['obj_pos'], dtype=np.float32)

        if mask_3d_volume.shape[-1] > 64:
            mask_3d_volume = mask_3d_volume[..., 63:-1]
            flow_3d_volume = flow_3d_volume[..., 63:-1]
            tsdf = tsdf[..., 63:-1]

        if self.mask_out_bg:
            pass

        for count_seq, step_id in enumerate(idx_seq):
            
            sample['%d-verts' % count_seq] = verts @ obj_rot[step_id-1, :3, :3].T + obj_pos[step_id-1]
            #sample['%d-verts' % count_seq] = verts
            sample['faces'] = faces

            sample['%d-gt_T' % count_seq] = gt_T[step_id]

            if self.use_velocity_action:
                cur_pos = gt_T[step_id][:3, 3]
                cur_pos[1] -= self.camera_translation
                sample['%d-action' % count_seq] = self.get_velocity_action(pos=cur_pos, vel=cur_vel[step_id - 1])
            else:
                sample['%d-action' % count_seq] = np.zeros_like(self.get_action([0, 0, 0]))   # There is no action in the dataset
            # (240, 320, 3)
            sample['%d-color_image' % count_seq] = self.padd_image(
                np.asarray(color[step_id - 1, ..., :3], dtype=np.uint8), size=(240, 320, 3), pad_value=0
            )

            sample['%d-depth_image' % count_seq] = self.padd_image(
                np.asarray(depth[step_id - 1], dtype=np.float32), size=(240, 320), pad_value=self.max_depth
            )

            '''color_heightmap = draw_arrow(
                height_map[step_id],
                (0, 0, 0)
            )
            sample['%d-color_heightmap' % count_seq] = color_heightmap'''
            sample['%d-color_heightmap' % count_seq] = height_map[step_id - 1]

            '''tsdf = self._get_volume(
                color_image=color[step_id, ..., :3].astype(np.uint8),
                depth_image=depth[step_id].astype(np.float32),
                cam_intr=camera_intrinsics[step_id],
                cam_pose=camera_pose[step_id],
                vol_bnds=view_bounds[step_id]
            )'''

            sample['%d-tsdf' % count_seq] = tsdf[step_id - 1]
            #sample['%d-tsdf' % count_seq] = np.zeros((128, 128, 128), dtype=np.float32)

            sample['%d-mask_3d' % count_seq] = mask_3d_volume[step_id]
            sample['%d-scene_flow_3d' % count_seq] = flow_3d_volume[step_id]

        return sample

    def __len__(self):
        return len(self.seq_paths)

    def _get_volume(self, color_image, depth_image, cam_intr, cam_pose, vol_bnds=None):
        # voxel_size = 0.004

        if vol_bnds is None:
            vol_bnds = np.array([[0.244, 0.756],
                                 [-0.256, 0.256],
                                 [0.0, 0.192]])
        self.tsdf_vol.integrate(color_image, depth_image, cam_intr, cam_pose, obs_weight=1.)
        volume = np.asarray(self.tsdf_vol.get_volume()[0])
        volume = np.transpose(volume, [1, 0, 2])
        return volume

    def _inverse_trans(self, T):

        inv_T = np.zeros_like(T)
        inv_T[:3, :3] = T[:3, :3].T
        inv_T[:3, 3:] = - inv_T[:3, :3] @ T[:3, 3:]
        inv_T[3, 3] = 1.0

        return inv_T

    def padd_image(self, img, size, pad_value=0):

        H_pad = (size[0] - img.shape[0]) // 2
        W_pad = (size[1] - img.shape[1]) // 2
        padded_img = np.ones(size, dtype=img.dtype) * pad_value
        padded_img[H_pad:H_pad + img.shape[0], W_pad:W_pad + img.shape[1]] = img

        return padded_img


    def get_heightmap(self, color_image, depth_image, cam_param):
        color_heightmap, depth_heightmap = utils.get_heightmap(
            color_img=color_image,
            depth_img=depth_image,
            cam_intrinsics=cam_param['camera_intr'],
            cam_pose=cam_param['camera_pose'],
            workspace_limits=self.view_bounds,
            heightmap_resolution=self._heightmap_pixel_size
        )
        return color_heightmap, depth_heightmap

    def _get_scene_flow_3d(self, gt_T):
        vol_bnds = self.view_bounds
        scene_flow = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds] + [3])
        mask = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds], dtype=np.int)

        cur_cnt = 0
        #import pdb
        #pdb.set_trace()
        for obj_id, old_po_or in zip(self.object_ids, old_po_ors):
            #position, orientation = p.getBasePositionAndOrientation(obj_id)
            new_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

            position, orientation = old_po_or
            old_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

            motion = new_coord - old_coord

            valid_idx = np.logical_and(
                np.logical_and(old_coord[:, 1] >= 0, old_coord[:, 1] < 128),
                np.logical_and(
                    np.logical_and(old_coord[:, 0] >= 0, old_coord[:, 0] < 128),
                    np.logical_and(old_coord[:, 2] >= 0, old_coord[:, 2] < 48)
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

    def get_action(self, action):
        direction, r, c = int(action[0]), int(action[1]), int(action[2])
        if direction < 0:
            direction += self.direction_num
        action_map = np.zeros(shape=[self.direction_num, self.volume_size[0], self.volume_size[1]], dtype=np.float32)
        action_map[direction, r, c] = 1

        return action_map

    def get_velocity_action(self, pos, vel):

        row, col = np.round(
            (pos[:2] - self.view_bounds[:2, 0]) * self.volume_size[0] / (self.view_bounds[:2, 1] - self.view_bounds[:2, 0])
        ).astype(np.int32)

        unit_vel = vel[:2] / np.linalg.norm(vel[:2], keepdims=True)
        theta = np.arctan2(unit_vel[1], unit_vel[0])

        direction = (theta >= np.linspace(-np.pi, np.pi, self.direction_num + 1)).argmin() - 1
        if direction < 0:
            direction = self.direction_num

        action_map = np.zeros(shape=[self.direction_num, self.volume_size[0], self.volume_size[1]], dtype=np.float32)
        action_map[direction, row, col] = 1

        return action_map

    @staticmethod
    def default_collate(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))
