import argparse
import time

import dataset_location
import losses
import torch
from model import SingleViewto3D
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh", "implicit", "parametric"], type=str
    )
    parser.add_argument("--n_patches", default=4, type=int,
                        help="parametric: number of 2D charts (n_points should be divisible by it)")
    parser.add_argument("--n_query", default=4096, type=int,
                        help="implicit: grid points sampled per shape per step (<=0 uses the full 32^3 grid)")
    parser.add_argument("--dataset", default=None, choices=["chair", "3c"], type=str,
                        help="overrides dataset_location.use_full_dataset")
    parser.add_argument("--exp_name", default=None, type=str,
                        help="checkpoint is saved as checkpoint_<exp_name>.pth (default: <type>)")
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type in ("vox", "implicit"):
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type in ("point", "parametric"):
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type in ("vox", "implicit"):
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type in ("point", "parametric"):
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def predict(model, images_gt, ground_truth_3d, args):
    """
    Forward pass, returning (prediction, matching ground truth).

    The implicit decoder is supervised on a random subset of the 32^3 grid per
    shape rather than all 32768 points, which would not fit in memory at batch 32.
    """
    if args.type != "implicit" or args.n_query <= 0:
        return model(images_gt, args), ground_truth_3d
    B = ground_truth_3d.shape[0]
    grid = model.decoder.grid.reshape(-1, 3)
    idx = torch.randint(0, grid.shape[0], (B, args.n_query), device=grid.device)
    logits = model(images_gt, args, query_points=grid[idx])
    occ = ground_truth_3d.reshape(B, -1).gather(1, idx)
    return logits, occ


def train_model(args):
    full = dataset_location.use_full_dataset if args.dataset is None else args.dataset == "3c"
    shapenet_path, r2n2_path, splits_path = dataset_location.get_paths(full)
    exp_name = args.exp_name or args.type
    r2n2_dataset = R2N2(
        "train",
        shapenet_path,
        r2n2_path,
        splits_path,
        return_voxels=args.type in ("vox", "implicit"),
        return_feats=args.load_feat,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{exp_name}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d, ground_truth_3d = predict(model, images_gt, ground_truth_3d, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0 and step > 0:
            print(f"Saving checkpoint at step {step}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_{exp_name}.pth",
            )

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    torch.save(
        {
            "step": args.max_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"checkpoint_{exp_name}.pth",
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
