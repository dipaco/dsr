import os

import numpy as np
import torch
import argparse
from tqdm import tqdm
import os.path as osp
from data import Data
from torch.utils.data import DataLoader
from model import ModelDSR
import itertools
from data import Data, ParabolicShotData
from data_shapestacks import ShapestacksDataset
from skimage import measure
#from midlevel_prediction.utils.visualization import write_obj, get_image_array_from_figure
from matplotlib import pyplot as plt
from chamfer_distance import ChamferDistance
import imageio
from data_shapestacks import ShapestacksDataset


parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, help='name of exp')
parser.add_argument('--log_dir', type=str, help='log dir.')
parser.add_argument('--resume', type=str, help='path to model')
parser.add_argument('--data_path', type=str, help='path to data')
parser.add_argument('--test_type', type=str, choices=['motion_visible', 'motion_full', 'mask_ordered', 'mask_unordered', 'full_model'])

parser.add_argument('--gpu', type=int, default=0, help='gpu id (single gpu)')
parser.add_argument('--object_num', type=int, default=5, help='number of objects')
parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
parser.add_argument('--batch', type=int, default=12, help='batch size')
parser.add_argument('--workers', type=int, default=2, help='number of workers in data loader')

parser.add_argument('--model_type', type=str, default='dsr', choices=['dsr', 'single', 'nowarp', 'gtwarp', '3dflow'])
parser.add_argument('--transform_type', type=str, default='se3euler', choices=['affine', 'se3euler', 'se3aa', 'se3spquat', 'se3quat'])

parser.add_argument('--dataset', type=str, help='Dataset type', choices=['dsr', 'throwing', 'shapestacks'], default='dsr')
parser.add_argument('--use_velocity_action', action='store_true', default=False)

# exp args
#parser.add_argument('--log_dir', type=str, help='log dir.')

import re
from torch._six import container_abcs, string_classes, int_classes
np_str_obj_array_pattern = re.compile(r'[SaUO]')

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

        out_dict = {key: default_collate([d[key] for d in batch]) for key in elem if (key != 'faces' and not key.endswith('verts'))}
        out_dict['faces'] = [d['faces'] for d in batch]

        nn = len([aa for aa in elem.keys() if aa.endswith('verts')])
        for ii in range(nn):
            out_dict[f'{ii}-verts'] = [d[f'{ii}-verts'] for d in batch]
        return out_dict
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    data, loaders = {}, {}
    for split in ['test']:

        if args.dataset == 'dsr':
            data[split] = Data(data_path=args.data_path, split=split, seq_len=args.seq_len)
        elif args.dataset == 'shapestacks':
            data[split] = ShapestacksDataset(base_path=args.data_path, split=split, num_objects=args.object_num - 1, sequence_length=args.seq_len, use_velocity_action=args.use_velocity_action)
            #view_bounds = np.array([[-0.56, 0.56], [-0.56, 0.56], [-0.56, 0.56]])
            view_bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [0.0, 5.0]], dtype=np.float32)
        else:
            data[split] = ParabolicShotData(data_path=args.data_path, split=split, seq_len=args.seq_len, use_velocity_action=args.use_velocity_action)
            aa = data[split].__getitem__(0)
            #view_bounds = np.array([[-0.56, 0.56], [-0.56, 0.56], [-0.56, 0.56]])
            view_bounds = np.array([[-1.05, 1.05], [-1.05, 1.05], [0, 1.05]])

        loaders[split] = DataLoader(dataset=data[split], batch_size=args.batch, num_workers=args.workers, collate_fn=default_collate)
    print('==> dataset loaded: [size] = {0}'.format(len(data['test'])))


    model = ModelDSR(
        object_num=args.object_num,
        transform_type=args.transform_type,
        motion_type='se3' if args.model_type != '3dflow' else 'conv',
        arch_type=args.dataset,
    )
    model.cuda()

    checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{args.gpu}'))
    model.load_state_dict(checkpoint['state_dict'])
    print('==> resume: ' + args.resume)

    with torch.no_grad():
        if args.test_type == 'motion_visible':
            evaluation_motion_visible(args, model, loaders['test'])
        
        if args.test_type == 'motion_full':
            evaluation_motion_full(args, model, loaders['test'])

        if args.test_type == 'mask_ordered':
            evaluation_mask_ordered(args, model, loaders['test'])

        if args.test_type == 'mask_unordered':
            evaluation_mask_unordered(args, model, loaders['test'])

        if args.test_type == 'full_model':
            evaluation_full_model(args, model, loaders['test'], view_bounds, args.dataset)

def evaluation_mask_unordered(args, model, loader):
    print(f'==> evaluation_mask (unordered)')
    iou_dict = [[] for _ in range(args.seq_len)]
    for batch in tqdm(loader):
        batch_size = batch['0-action'].size(0)
        last_s = model.get_init_repr(batch_size).cuda()
        logit_pred_list, mask_gt_list = [], []
        for step_id in range(args.seq_len):
            output = model(
                input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp'
            )
            if not args.model_type == 'single':
                last_s = output['s'].data

            logit_pred = output['init_logit']
            mask_gt = batch['%d-mask_3d' % step_id].cuda()
            iou_unordered = calc_iou_unordered(logit_pred, mask_gt)
            iou_dict[step_id].append(iou_unordered)
    print('mask_unordered (IoU) = ', np.mean([np.mean(np.concatenate(iou_dict[i])) for i in range(args.seq_len)]))


def calc_iou_unordered(logit_pred, mask_gt_argmax):
    # logit_pred: [B, K, S1, S2, S3], softmax, the last channel is empty
    # mask_gt_argmax: [B, S1, S2, S3], 0 represents empty
    B, K, S1, S2, S3 = logit_pred.size()
    logit_pred_argmax = torch.argmax(logit_pred, dim=1, keepdim=True)
    mask_gt_argmax = torch.unsqueeze(mask_gt_argmax, 1)
    mask_pred_onehot = torch.zeros_like(logit_pred).scatter(1, logit_pred_argmax, 1)[:, :-1]
    mask_gt_onehot = torch.zeros_like(logit_pred).scatter(1, mask_gt_argmax, 1)[:, 1:]
    K -= 1
    info_dict = {'I': np.zeros([B, K, K]), 'U': np.zeros([B, K, K])}
    for b in range(B):
        for i in range(K):
            for j in range(K):
                mask_gt = mask_gt_onehot[b, i]
                mask_pred = mask_pred_onehot[b, j]
                I = torch.sum(mask_gt * mask_pred).item()
                U = torch.sum(mask_gt + mask_pred).item() - I
                info_dict['I'][b, i, j] = I
                info_dict['U'][b, i, j] = U
    batch_ious = []
    for b in range(B):
        best_iou, best_p = 0, None
        for p in list(itertools.permutations(range(K))):
            cur_I = [info_dict['I'][b, i, p[i]] for i in range(K)]
            cur_U = [info_dict['U'][b, i, p[i]] for i in range(K)]
            cur_iou = np.mean(np.array(cur_I) / np.maximum(np.array(cur_U), 1))
            if cur_iou > best_iou:
                best_iou = cur_iou
        batch_ious.append(best_iou)

    return np.array(batch_ious)


def evaluation_full_model(args, model, loader, view_bounds, dataset='dsr'):
    print(f'==> evaluation_mask (ordered)')
    iou_dict = []
    iou_dict_per_step = []
    cd_metric = []

    results_path = os.path.join(args.log_dir, args.exp, 'results')
    os.makedirs(results_path, exist_ok=True)
    results_meshes_path = os.path.join(results_path, 'meshes')
    os.makedirs(results_meshes_path, exist_ok=True)
    results_images_path = os.path.join(results_path, 'images')
    os.makedirs(results_images_path, exist_ok=True)

    chamfer_dist_func = ChamferDistance()

    for batch_count, batch in enumerate(tqdm(loader)):

        if batch_count > 100:
            break

        batch_size = batch['0-action'].size(0)
        last_s = model.get_init_repr(batch_size).cuda()
        logit_pred_list, mask_gt_list = [], []
        for step_id in range(args.seq_len):

            if step_id >= 2:

                #logit_pred_argmax = torch.argmax(logit_pred, dim=1, keepdim=True)
                #mask_last_step = torch.zeros_like(logit_pred).scatter(1, logit_pred_argmax, 1)[:, :-1][:, 0]

                next_mask = (next_mask[:, 0, ...] > 0.8).float()

                '''vol_coords = torch.stack(torch.meshgrid(
                    torch.linspace(view_bounds[0, 0], view_bounds[0, 1], mask_last_step.shape[-3]),
                    torch.linspace(view_bounds[1, 0], view_bounds[1, 1], mask_last_step.shape[-2]),
                    torch.linspace(view_bounds[2, 0], view_bounds[2, 1], mask_last_step.shape[-1]),
                ), dim=-1)[None].repeat(batch_size, 1, 1, 1, 1).to(mask_last_step)
                vol_coords_shape = vol_coords.shape
                vol_coords = vol_coords.reshape(batch_size, -1, 3)

                eroded_mask = torch.nn.functional.conv3d(mask_last_step[:, None, ...], torch.ones((1, 1, 3, 3, 3), device=mask_last_step.device), padding=1) >= 27.0
                surface_mask = torch.logical_xor(eroded_mask[:, 0], mask_last_step)

                input_tsdf = []
                for vc, sm, em in zip(torch.split(vol_coords, 1, dim=0), torch.split(surface_mask, 1, dim=0), torch.split(eroded_mask, 1, dim=0)):

                    dist1, dist2 = chamfer_dist_func(vc, vc[:, sm.reshape(-1).bool(), :])

                    dist1 = dist1.sqrt()
                    dist1[:, em.reshape(-1).bool()] *= -1.0
                    dist1 = dist1.reshape(1, *vol_coords_shape[1:-1])

                    #import pdb
                    #pdb.set_trace()

                    dist1 = torch.clamp(dist1, min=-1.0, max=1.0)

                    input_tsdf.append(dist1)

                input_tsdf = torch.cat(input_tsdf, dim=0).unsqueeze(1)



                #2.05 / 128 * 2

                plt.imshow(batch['%d-tsdf' % step_id].cuda().unsqueeze(1)[0, 0, :, :, 32].cpu().numpy(), interpolation=None)
                plt.colorbar()
                plt.savefig('/home/diegopc/tmp/dsr_pred.png')

                plt.figure()
                plt.imshow(input_tsdf[0, 0, :, :, 32].cpu().numpy(), interpolation=None)
                plt.colorbar()
                plt.savefig('/home/diegopc/tmp/dsr_cd.png')

                plt.figure()
                plt.imshow(surface_mask[0, :, :, 32].cpu().numpy(), interpolation=None)
                plt.colorbar()
                plt.savefig('/home/diegopc/tmp/dsr_surf.png')

                input_tsdf = -2.0 * mask_last_step + 1
                plt.figure()
                plt.imshow(input_tsdf[0, :, :, 32].cpu().numpy(), interpolation=None)
                plt.colorbar()
                plt.savefig('/home/diegopc/tmp/dsr_alt.png')

                #import pdb
                #pdb.set_trace()'''


                input_tsdf = (-2.0 * next_mask + 1).unsqueeze(1)

                #input_tsdf = input_tsdf.unsqueeze(1)
            else:
                input_tsdf = batch['%d-tsdf' % step_id].cuda().unsqueeze(1)

            output = model(
                input_volume=input_tsdf,
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp',
                next_mask=True,
            )
            if not args.model_type == 'single':
                last_s = output['s'].data

            logit_pred = output['init_logit']
            next_mask = output['next_mask']
            mask_gt = batch['%d-mask_3d' % step_id].cuda()
            logit_pred_list.append(logit_pred)
            mask_gt_list.append(mask_gt)
        iou_ordered, iou_ordered_per_step, cd_metric_batch = calc_iou_full_model(batch, logit_pred_list, mask_gt_list, batch_count, batch_size, results_meshes_path, results_images_path, view_bounds, dataset)

        iou_dict.append(iou_ordered)
        iou_dict_per_step.append(iou_ordered_per_step)
        cd_metric.append(cd_metric_batch)

    print('mask_ordered (IoU) = ', np.mean(np.concatenate(iou_dict)), np.std(np.concatenate(iou_dict)))
    print('mask_ordered per step (IoU) = ', np.mean(np.concatenate(iou_dict_per_step), axis=0), np.std(np.concatenate(iou_dict_per_step), axis=0))
    if dataset == 'shapestacks':
        print('Chamfer Distance (CD) = ', np.mean(np.concatenate(cd_metric, axis=1)), np.std(np.concatenate(cd_metric, axis=1)))
        print('Chamfer Distance per_step (CD) = ', np.mean(np.concatenate(cd_metric, axis=1), axis=(1, 2)), np.std(np.concatenate(cd_metric, axis=1), axis=(1, 2)))
    else:
        print('Chamfer Distance (CD) = ', np.mean(np.concatenate(cd_metric, axis=1)), np.std(np.concatenate(cd_metric, axis=1)))
        print('Chamfer Distance per_step (CD) = ', np.mean(np.concatenate(cd_metric, axis=1), axis=1), np.std(np.concatenate(cd_metric, axis=1), axis=1))

    stats_path = os.path.join(results_path, 'stats_IoU.pt')

    torch.save(
        {
            'IoU': torch.from_numpy(np.concatenate(iou_dict_per_step)),
            'CD': torch.from_numpy(np.concatenate(cd_metric, axis=1)),
        },
        stats_path
    )


def calc_iou_full_model(batch, logit_pred_list, mask_gt_argmax_list, batch_count, batch_size, results_path, results_images_path, view_bounds, dataset='dsr'):

    # logit_pred_list: [L, B, K, S1, S2, S3], softmax, the last channel is empty
    # mask_gt_argmax_list: [L, B, S1, S2, S3], 0 represents empty
    chamfer_dist_func = ChamferDistance()
    L = len(logit_pred_list)
    B, K, S1, S2, S3 = logit_pred_list[0].size()
    K -= 1
    unit_sphere_scale = 1.7071 / 4
    info_dict = {'I': np.zeros([L, B, K]), 'U': np.zeros([L, B, K]), 'CD': np.zeros([L, B, K])}
    for l in range(L):
        logit_pred = logit_pred_list[l]
        mask_gt_argmax = mask_gt_argmax_list[l]
        logit_pred_argmax = torch.argmax(logit_pred, dim=1, keepdim=True)
        mask_gt_argmax = torch.unsqueeze(mask_gt_argmax, 1)
        mask_pred_onehot = torch.zeros_like(logit_pred).scatter(1, logit_pred_argmax, 1)[:, :-1]
        mask_gt_onehot = torch.zeros_like(logit_pred).scatter(1, mask_gt_argmax, 1)[:, 1:]
        for b in range(B):
            example_folder = os.path.join(results_path, f'mesh_{b + batch_count * batch_size}')
            os.makedirs(example_folder, exist_ok=True)
            for i in range(K):

                # save the rgb images
                rgb_image = batch[f'{l}-color_image'][b].cpu().numpy()
                imageio.imwrite(os.path.join(results_images_path, f'gt_frame_{l + 1}_ob_{i}.png'), rgb_image)


                mask_gt = mask_gt_onehot[b, i]
                mask_pred = mask_pred_onehot[b, i]

                if dataset == 'shapestacks':
                    gt_verts = unit_sphere_scale * batch[f'{l}-verts'][b][i]  # .cpu().numpy()
                    gt_faces = batch[f'faces'][b][i]  # .cpu().numpy()
                elif dataset == 'dsr':
                    pass
                elif dataset == 'throwing':
                    gt_verts = unit_sphere_scale * batch[f'{l}-verts'][b]  # .cpu().numpy()
                    gt_faces = batch[f'faces'][b]  # .cpu().numpy()

                write_obj(os.path.join(example_folder, f'gt_frame_{l + 1}_ob_{i}.obj'), gt_verts, gt_faces + 1)

                #gt_verts, gt_faces = compute_mesh(
                #    mask_gt.cpu().numpy(),
                #    view_bounds,
                #)
                #write_obj(os.path.join(results_path, f'gt_mesh_{b + batch_count * batch_size}_frame_{l + 1}_ob_{i}.obj'), verts=verts, faces=faces + 1)
                pred_verts, pred_faces = compute_mesh(
                    mask_pred.cpu().numpy().transpose(1, 0, 2),
                    view_bounds,
                )

                if dataset == 'shapestacks':
                    mujoco_transform = np.zeros((3, 3))
                    mujoco_transform[0, 0] = 1.
                    mujoco_transform[1, 2] = -1.
                    mujoco_transform[2, 1] = 1.
                    pred_verts = pred_verts @ mujoco_transform

                pred_verts = unit_sphere_scale * pred_verts
                write_obj(os.path.join(example_folder, f'pred_frame_{l + 1}_ob_{i}.obj'), verts=pred_verts, faces=pred_faces + 1)

                info_dict['CD'][l, b, i] = compute_chamfer_distance(pred_verts, pred_faces, gt_verts, gt_faces, chamfer_dist_func)

                I = torch.sum(mask_gt * mask_pred).item()
                U = torch.sum(mask_gt + mask_pred).item() - I
                info_dict['I'][l, b, i] = I
                info_dict['U'][l, b, i] = U

    batch_ious = []
    for b in range(B):
        best_iou, best_p = 0, None
        #for p in list(itertools.permutations(range(K))):
        #for p in list(itertools.permutations(range(K))):
        cur_I = [info_dict['I'][l, b, i] for l in range(L) for i in range(K)]
        cur_U = [info_dict['U'][l, b, i] for l in range(L) for i in range(K)]

        cur_iou = np.array(cur_I) / np.maximum(np.array(cur_U), 1)
        #if cur_iou > best_iou:
        #    best_iou = cur_iou
        batch_ious.append(cur_iou)

    batch_ious = np.array(batch_ious)
    return batch_ious.mean(axis=-1), batch_ious, info_dict['CD']


def compute_chamfer_distance(pred_verts, pred_faces, gt_verts, gt_faces, chamfer_dist_func=ChamferDistance(), num_points=10000):

    pred_surf_points = sample_surface_points(pred_verts, pred_faces, num_points)
    gt_surf_points = sample_surface_points(gt_verts, gt_faces, num_points)

    dist1, dist2 = chamfer_dist_func(
        torch.from_numpy(pred_surf_points[None]).cuda().float(),
        torch.from_numpy(gt_surf_points[None]).cuda().float()
    )

    return dist1.mean() + dist2.mean()


def compute_mesh(occ_vol, view_bounds):
    scale = (view_bounds[:, 1] - view_bounds[:, 0]).max()
    try:
        verts, faces, _, _ = measure.marching_cubes_lewiner(-occ_vol + 0.5, 0)
    except:
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).astype(np.float32)
        faces = np.array([[0, 1, 2]]).astype(np.int32)

    #verts = verts / (occ_vol.shape[0] - 1.0) - 0.5
    verts = verts / (occ_vol.shape[0] - 1.0)
    verts = scale * verts + view_bounds[:, 0]

    return verts, faces


def sample_surface_points(verts, faces, num_points):

    triangles = verts[faces]
    seg1 = triangles[:, 1] - triangles[:, 0]
    seg2 = triangles[:, 2] - triangles[:, 1]
    face_areas = np.linalg.norm(np.cross(seg1, seg2), axis=-1) / 2
    idx = np.random.choice(face_areas.shape[0], size=num_points, replace=True, p=face_areas / face_areas.sum())
    weights = np.random.rand(num_points, 3)
    weights /= weights.sum(axis=-1, keepdims=True)

    sample_points = (triangles[idx] * weights[..., None]).sum(axis=-2)

    return sample_points


def evaluation_motion_visible(args, model, loader):
    print('==> evaluation_motion (visible surface)')
    mse_dict = [0 for _ in range(args.seq_len)]
    data_num = 0
    for batch in tqdm(loader):
        batch_size = batch['0-action'].size(0)
        data_num += batch_size
        last_s = model.get_init_repr(batch_size).cuda()
        for step_id in range(args.seq_len):
            output = model(
                input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp'
            )
            if not args.model_type in ['single', '3dflow'] :
                last_s = output['s'].data

            tsdf = batch['%d-tsdf' % step_id].cuda().unsqueeze(1)
            mask = batch['%d-mask_3d' % step_id].cuda().unsqueeze(1)
            surface_mask = ((tsdf > -0.99).float()) * ((tsdf < 0).float()) * ((mask > 0).float())
            surface_mask[..., 0] = 0

            target = batch['%d-scene_flow_3d' % step_id].cuda()
            pred = output['motion']

            mse = torch.sum((target - pred) ** 2 * surface_mask, dim=[1, 2, 3, 4]) / torch.sum(surface_mask, dim=[1, 2, 3, 4])
            mse_dict[step_id] += torch.sum(mse).item() * 0.16
            # 0.16(0.4^2) is the scale to convert the unit from "voxel" to "cm".
            # The voxel size is 0.4cm. Here we use seuqre error.
    print('motion_visible (MSE in cm) = ', np.mean([np.mean(mse_dict[i]) / data_num for i in range(args.seq_len)]))


def evaluation_motion_full(args, model, loader):
    print('==> evaluation_motion (full volume)')
    mse_dict = [0 for _ in range(args.seq_len)]
    data_num = 0
    results_path = os.path.join(args.log_dir, args.exp, 'results')
    os.makedirs(results_path, exist_ok=True)
    for batch_count, batch in enumerate(tqdm(loader)):

        batch_size = batch['0-action'].size(0)
        data_num += batch_size
        last_s = model.get_init_repr(batch_size).cuda()
        for step_id in range(args.seq_len):

            if step_id >= 2:
                next_mask = (next_mask[:, 0, ...] > 0.8).float()
                input_tsdf = (-2.0 * next_mask + 1).unsqueeze(1)
            else:
                input_tsdf = batch['%d-tsdf' % step_id].cuda().unsqueeze(1)

            output = model(
                input_volume=input_tsdf,
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp',
                next_mask=True,
            )
            if not args.model_type in ['single', '3dflow'] :
                last_s = output['s'].data

            target = batch['%d-scene_flow_3d' % step_id].cuda()
            next_mask = output['next_mask']
            pred = output['motion']

            mse = torch.mean((target - pred) ** 2, dim=[1, 2, 3, 4])
            mse_dict[step_id] += torch.sum(mse).item() * 0.16
            # 0.16(0.4^2) is the scale to convert the unit from "voxel" to "cm".
            # The voxel size is 0.4cm. Here we use seuqre error.
    print('motion_full (MSE in cm) = ', np.mean([np.mean(mse_dict[i]) / data_num for i in range(args.seq_len)]), np.std([np.mean(mse_dict[i]) / data_num for i in range(args.seq_len)]))

    stats_path = os.path.join(results_path, 'stats_mse.pt')

    torch.save(
        {
            'mse': torch.tensor([np.mean(mse_dict[i]) / data_num for i in range(args.seq_len)]),
        },
        stats_path
    )


def write_obj(filepath, verts, faces):
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in verts:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for fc in faces:
            f.write(f"f {fc[0]:d} {fc[1]:d} {fc[2]:d}\n")


if __name__ == '__main__':
    main()