import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import json
import math
import random
import numpy as np
import os
import imageio
import utils_viz
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh', 'implicit', 'parametric'], type=str)
    parser.add_argument('--n_patches', default=4, type=int)
    parser.add_argument('--dataset', default=None, choices=['chair', '3c'], type=str,
                        help='overrides dataset_location.use_full_dataset')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='loads checkpoint_<exp_name>.pth (default: <type>)')
    parser.add_argument('--eval_class', default='all', choices=['all', 'chair', 'car', 'plane'], type=str,
                        help='restrict the test split to one class')
    parser.add_argument('--seed', default=0, type=int,
                        help='fixes which view of each test model is drawn, so runs are comparable')
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--num_views', default=36, type=int)
    parser.add_argument('--vis_dir', default='vis', type=str)
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.exp_name}')
    plt.savefig(f'eval_{args.exp_name}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def vox_to_mesh(voxels):
    """
    Marching-cubes a predicted grid and move it into the gt mesh's frame.

    Same chain evaluate() applies to its sampled points: index space -> Mem2Ref,
    rotate pi about y, recenter. Returns (verts, faces) or None when the grid has
    no surface at the 0.5 level.
    """
    H, W, D = voxels.shape[2:]
    grid = torch.sigmoid(voxels).detach().cpu().squeeze().numpy()
    verts, faces = mcubes.marching_cubes(grid, isovalue=0.5)
    if len(verts) == 0 or len(faces) == 0:
        return None

    verts = torch.tensor(verts).float().unsqueeze(0)
    verts = utils_vox.Mem2Ref(verts, H, W, D)
    Rot = axis_angle_to_matrix(torch.as_tensor(np.array([[0.0, -math.pi, 0.0]])).float())
    verts = Rotate(Rot).transform_points(verts)
    verts = verts - verts.mean(1, keepdim=True)
    return verts[0], torch.tensor(faces.astype(int))


def render_strip(verts, faces, color, scale, args):
    """Centers, rescales and orbits one mesh; blank frames if it has no surface."""
    if verts is None:
        return utils_viz.blank_frames(args.image_size, args.num_views)
    verts = (verts - verts.mean(dim=0, keepdim=True)) / scale
    textures = pytorch3d.renderer.TexturesVertex(
        torch.tensor(color).expand(verts.shape).unsqueeze(0)
    )
    mesh = pytorch3d.structures.Meshes(
        [verts], [faces], textures=textures.to(args.device)
    ).to(args.device)
    return utils_viz.render_360(
        mesh, image_size=args.image_size, num_views=args.num_views,
        dist=3.0, device=args.device,
    )


def visualize_prediction(images_gt, predictions, mesh_gt, step, args):
    """Writes an [input RGB | prediction | ground truth mesh] gif and still."""
    os.makedirs(args.vis_dir, exist_ok=True)

    gt_verts = mesh_gt.verts_list()[0].detach().cpu()
    gt_faces = mesh_gt.faces_list()[0].detach().cpu()
    scale = (gt_verts - gt_verts.mean(0, keepdim=True)).norm(dim=1).max()

    rgb = (images_gt[0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    rgb = np.array(Image.fromarray(rgb).resize((args.image_size, args.image_size)))
    rgb_frames = [rgb] * args.num_views

    if args.type in ("point", "parametric"):
        pts = predictions[0].detach()
        pts = (pts - pts.mean(0, keepdim=True)) / scale.to(pts.device)
        pred_frames = utils_viz.render_points_360(
            pts, image_size=args.image_size, num_views=args.num_views,
            dist=3.0, device=args.device,
        )
    else:
        if args.type == "mesh":
            pred = (predictions.verts_list()[0].detach().cpu(), predictions.faces_list()[0].detach().cpu())
        else:
            pred = vox_to_mesh(predictions)
        pred_frames = render_strip(
            *(pred if pred is not None else (None, None)),
            color=(0.85, 0.45, 0.35), scale=scale, args=args,
        )
    gt_frames = render_strip(
        gt_verts, gt_faces, color=(0.45, 0.55, 0.85), scale=scale, args=args,
    )

    strip = [np.concatenate(f, axis=1) for f in zip(rgb_frames, pred_frames, gt_frames)]
    imageio.mimsave(f'{args.vis_dir}/{step}_{args.exp_name}.gif', strip, duration=1000 // 15, loop=0)
    plt.imsave(f'{args.vis_dir}/{step}_{args.exp_name}.png', strip[0])
    print(f'wrote {args.vis_dir}/{step}_{args.exp_name}.gif')


def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type in ("vox", "implicit"):
        voxels_src = torch.sigmoid(predictions)
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        if len(vertices_src) == 0 or len(faces_src) == 0:
            # Nothing crosses 0.5: score it as a miss rather than crashing.
            gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
            pred_points = torch.full_like(gt_points, 1e3)
            return compute_sampling_metrics(pred_points, gt_points, thresholds)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type in ("point", "parametric"):
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type in ("vox", "implicit"):
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.exp_name = args.exp_name or args.type
    full = dataset_location.use_full_dataset if args.dataset is None else args.dataset == '3c'
    shapenet_path, r2n2_path, splits_path = dataset_location.get_paths(full)
    r2n2_dataset = R2N2("test", shapenet_path, r2n2_path, splits_path, return_voxels=True, return_feats=args.load_feat)
    if args.eval_class != 'all':
        synset = dataset_location.CLASS_SYNSETS[args.eval_class]
        if synset not in r2n2_dataset.synset_start_idxs:
            raise ValueError(f'{args.eval_class} is not in {splits_path}')
        start = r2n2_dataset.synset_start_idxs[synset]
        r2n2_dataset = torch.utils.data.Subset(
            r2n2_dataset, range(start, start + r2n2_dataset.synset_num_models[synset]))

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []
    f1_by_class = {}

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.exp_name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        with torch.no_grad():
            predictions = model(images_gt, args)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        if (step % args.vis_freq) == 0:
            visualize_prediction(feed_dict['images'], predictions, mesh_gt, step, args)


        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        for synset, f1 in zip(feed_dict['synset_id'], f1_05.tolist()):
            f1_by_class.setdefault(dataset_location.SYNSET_CLASSES.get(synset, synset), []).append(f1)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)

    summary = {
        'exp_name': args.exp_name,
        'eval_class': args.eval_class,
        'thresholds': thresholds,
        'f1': avg_f1_score.tolist(),
        'precision': torch.stack(avg_p_score).mean(0).tolist(),
        'recall': torch.stack(avg_r_score).mean(0).tolist(),
        'f1@0.05_by_class': {c: float(np.mean(v)) for c, v in f1_by_class.items()},
        'n_by_class': {c: len(v) for c, v in f1_by_class.items()},
    }
    for c, v in summary['f1@0.05_by_class'].items():
        print(f'{c:>8s}: F1@0.05 = {v:.3f}  (n={summary["n_by_class"][c]})')
    with open(f'eval_{args.exp_name}_{args.eval_class}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
