from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d


def make_grid(res=32):
   
    lin = torch.linspace(-1.0, 1.0, res)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)


class ImplicitDecoder(nn.Module):

    def __init__(self, feat_dim=512, hidden=256, n_freqs=6, res=32):
        super().__init__()
        self.res = res
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freqs)) * torch.pi, persistent=False)
        self.register_buffer("grid", make_grid(res), persistent=False)
        pe_dim = 3 + 3 * 2 * n_freqs

        self.point_in = nn.Linear(pe_dim, hidden)
        self.feat_in = nn.Linear(feat_dim, hidden)
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
        )
        self.skip_point = nn.Linear(pe_dim, hidden)
        self.skip_feat = nn.Linear(feat_dim, hidden)
        self.block2 = nn.Sequential(
            nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True), nn.Linear(hidden, 1),
        )

    def encode_points(self, xyz):
        angles = xyz.unsqueeze(-1) * self.freqs  # b x N x 3 x F
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return torch.cat([xyz, pe], dim=-1)

    def query(self, feat, xyz):
        
        pe = self.encode_points(xyz)
        h = self.point_in(pe) + self.feat_in(feat).unsqueeze(1)
        h = self.block1(h)
        h = h + self.skip_point(pe) + self.skip_feat(feat).unsqueeze(1)
        return self.block2(h).squeeze(-1)

    def forward(self, feat, xyz=None):

        if xyz is not None:
            return self.query(feat, xyz)
        B, r = feat.shape[0], self.res
        xyz = self.grid.reshape(1, -1, 3).expand(B, -1, -1)
        return self.query(feat, xyz).reshape(B, 1, r, r, r)


class ParametricDecoder(nn.Module):


    def __init__(self, feat_dim=512, hidden=256, n_patches=4):
        super().__init__()
        self.n_patches = n_patches
        self.uv_in = nn.ModuleList([nn.Linear(2, hidden) for _ in range(n_patches)])
        self.feat_in = nn.ModuleList([nn.Linear(feat_dim, hidden) for _ in range(n_patches)])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True), nn.Linear(hidden, 3),
            )
            for _ in range(n_patches)
        ])

    def forward(self, feat, n_points=None, uv=None):
        if uv is None:
            M = n_points // self.n_patches
            uv = torch.rand(feat.shape[0], self.n_patches, M, 2, device=feat.device)
        points = []
        for k in range(self.n_patches):
            h = self.uv_in[k](uv[:, k]) + self.feat_in[k](feat).unsqueeze(1)
            points.append(self.mlps[k](h))
        return torch.cat(points, dim=1)


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (128, 2, 2, 2)),
                nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 1, 3, stride=1, padding=1),
            )
        elif args.type == "implicit":
            # Input: b x 512 (+ b x N x 3 query points)
            # Output: b x N logits, or b x 1 x 32 x 32 x 32 over the full grid
            self.decoder = ImplicitDecoder(feat_dim=512)
        elif args.type == "parametric":
            # Input: b x 512 (+ uv samples in [0,1]^2 per patch)
            # Output: b x args.n_points x 3
            self.n_point = args.n_points
            self.decoder = ParametricDecoder(feat_dim=512, n_patches=args.n_patches)
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.n_point * 3),
            )
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.n_vert = mesh_pred.verts_packed().shape[0]
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.n_vert * 3),
            )

    def forward(self, images, args, query_points=None):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif args.type == "implicit":
            return self.decoder(encoded_feat, query_points)

        elif args.type == "parametric":
            return self.decoder(encoded_feat, n_points=self.n_point)

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat).reshape(B, self.n_point, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat).reshape(B, self.n_vert, 3)

            if len(self.mesh_pred) != B:
                src = ico_sphere(4, self.device)
                self.mesh_pred = pytorch3d.structures.Meshes(
                    src.verts_list()*B, src.faces_list()*B
                )
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

