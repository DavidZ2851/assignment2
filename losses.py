import torch
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grids
	loss = torch.nn.functional.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	
	dist_src = knn_points(point_cloud_src, point_cloud_tgt, K=1).dists
	dist_tgt = knn_points(point_cloud_tgt, point_cloud_src, K=1).dists
	
	loss_chamfer = dist_src.squeeze(-1).mean(dim=1) + dist_tgt.squeeze(-1).mean(dim=1)
	return loss_chamfer.mean()
	# implement chamfer loss from scratch

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	# implement laplacian smoothening loss
	return loss_laplacian