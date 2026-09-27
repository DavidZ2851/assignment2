import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
import utils_viz






def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--output_dir', default='data', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--num_views', default=36, type=int)
    return parser

def visualize_voxels(voxels_src, voxels_tgt, args):
    """
    Renders the optimized voxel grid beside the ground truth as one gif.

    Left half is the fit, right half is the target, both orbited by the same
    camera so the two are directly comparable frame for frame.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # voxels_src are logits, so squash them before thresholding -- that way the
    # 0.5 iso-level means "occupied" for the fit and the target alike.
    meshes = {
        "optimized": utils_viz.voxels_to_mesh(
            torch.sigmoid(voxels_src), color=(0.85, 0.45, 0.35), device=args.device
        ),
        "ground truth": utils_viz.voxels_to_mesh(
            voxels_tgt, color=(0.45, 0.55, 0.85), device=args.device
        ),
    }

    frames = {}
    for name, mesh in meshes.items():
        if mesh is None:
            print(f"warning: {name} voxel grid has no surface at the 0.5 level set")
            frames[name] = utils_viz.blank_frames(args.image_size, args.num_views)
        else:
            frames[name] = utils_viz.render_360(
                mesh, image_size=args.image_size, num_views=args.num_views,
                dist=3.0, device=args.device,
            )

    utils_viz.save_side_by_side_gif(
        frames["optimized"], frames["ground truth"],
        os.path.join(args.output_dir, "q1.1_voxel_fit.gif"),
    )


def visualize_pointclouds(pointclouds_src, pointclouds_tgt, args):
    """
    Renders the optimized point cloud beside the ground truth as one gif.

    Both clouds are normalized by the SAME transform, derived from the target,
    so the view is framed sensibly without hiding any residual offset or scale
    error in the fit.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    center, scale = utils_viz.unit_sphere_transform(pointclouds_tgt)
    normalize = lambda p: (p.detach() - center) / scale

    frames_src = utils_viz.render_points_360(
        normalize(pointclouds_src), color=(0.85, 0.45, 0.35),
        image_size=args.image_size, num_views=args.num_views, device=args.device,
    )
    frames_tgt = utils_viz.render_points_360(
        normalize(pointclouds_tgt), color=(0.45, 0.55, 0.85),
        image_size=args.image_size, num_views=args.num_views, device=args.device,
    )
    utils_viz.save_side_by_side_gif(
        frames_src, frames_tgt,
        os.path.join(args.output_dir, "q1.2_point_fit.gif"),
    )


def visualize_meshes(mesh_src, mesh_tgt, args):
    """
    Renders the optimized mesh beside the ground truth as one gif.

    Both meshes share one normalization derived from the target, so the fit is
    not silently re-centered or re-scaled onto it.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    center, scale = utils_viz.unit_sphere_transform(mesh_tgt.verts_list()[0])
    frames = []
    for mesh, color in ((mesh_src, (0.85, 0.45, 0.35)), (mesh_tgt, (0.45, 0.55, 0.85))):
        prepared = utils_viz.prepare_mesh(
            mesh, color=color, center=center, scale=scale, device=args.device
        )
        frames.append(utils_viz.render_360(
            prepared, image_size=args.image_size, num_views=args.num_views,
            dist=3.0, device=args.device,
        ))

    utils_viz.save_side_by_side_gif(
        frames[0], frames[1], os.path.join(args.output_dir, "q1.3_mesh_fit.gif"),
    )


def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)

        # visualization
        visualize_voxels(voxels_src, voxels_tgt, args)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        

        # visualization
        visualize_pointclouds(pointclouds_src, pointclouds_tgt, args)
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)        

        # visualization (fit_mesh deforms mesh_src in place)
        visualize_meshes(mesh_src, mesh_tgt, args)


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
