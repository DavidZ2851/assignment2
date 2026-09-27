"""
Rendering helpers for turning fitted 3D outputs into 360-degree gifs.

Mirrors the renderer setup from assignment 1 (`starter/utils.py` and
`starter/5_3_implicit_surfaces.py`): a voxel grid becomes a mesh via marching
cubes, the mesh is orbited by a camera, and the frames are written out as a gif.
"""
import imageio
import mcubes
import numpy as np
import pytorch3d
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_mesh_renderer(image_size=256, lights=None, device=None):
    if device is None:
        device = get_device()
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )


def voxels_to_mesh(voxels, isovalue=0.5, color=(0.5, 0.6, 0.9), device=None):
    """
    Marching-cubes a single occupancy grid into a pytorch3d mesh.

    Args:
        voxels: occupancy *probabilities* in [0, 1], shaped (D, H, W) with any
            number of leading singleton dims. Logits must be passed through
            torch.sigmoid first, otherwise `isovalue` means nothing.
        isovalue: level set to extract; 0.5 is the decision boundary of a
            sigmoid, i.e. the surface between occupied and empty.

    Returns:
        A Meshes object in a [-1, 1]^3 box, or None when the grid is entirely
        occupied or entirely empty (marching cubes finds no surface to stitch).
    """
    if device is None:
        device = get_device()

    grid = voxels.detach().squeeze().cpu().numpy().astype(np.float32)
    assert grid.ndim == 3, f"expected a single 3D grid, got shape {grid.shape}"

    vertices, faces = mcubes.marching_cubes(grid, isovalue=isovalue)
    if len(vertices) == 0 or len(faces) == 0:
        return None

    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Voxel indices -> a centered [-1, 1] box, so the two grids we compare are
    # rendered at the same scale regardless of their resolution.
    resolution = max(grid.shape)
    vertices = vertices / (resolution - 1) * 2 - 1

    textures = TexturesVertex(torch.tensor(color).expand(vertices.shape).unsqueeze(0))
    return pytorch3d.structures.Meshes(
        [vertices], [faces], textures=textures.to(device)
    ).to(device)


def render_360(mesh, image_size=256, dist=3.0, elev=30.0, num_views=36, device=None):
    """Orbits a camera around `mesh` and returns the frames as uint8 arrays."""
    if device is None:
        device = get_device()
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    frames = []
    for azim in torch.linspace(0, 360, num_views + 1)[:-1]:
        R, T = look_at_view_transform(
            dist=dist, elev=elev, azim=float(azim), device=device
        )
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        # Light rides with the camera, otherwise half the orbit sits in shadow.
        lights = PointLights(location=cameras.get_camera_center(), device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        frames.append((rend * 255).astype(np.uint8))
    return frames


def blank_frames(image_size, num_views):
    """White stand-in frames, used when a grid has no surface to render."""
    return [np.full((image_size, image_size, 3), 255, dtype=np.uint8)] * num_views


def save_gif(frames, output_path, fps=15):
    imageio.mimsave(output_path, frames, duration=1000 // fps, loop=0)
    print(f"wrote {output_path}")


def save_side_by_side_gif(frames_left, frames_right, output_path, fps=15):
    """Stitches two equal-length frame lists into one left|right gif."""
    combined = [np.concatenate([l, r], axis=1) for l, r in zip(frames_left, frames_right)]
    save_gif(combined, output_path, fps=fps)


# ---------------------------------------------------------------- point clouds

def get_points_renderer(
    image_size=256, device=None, radius=0.012, background_color=(1, 1, 1)
):
    if device is None:
        device = get_device()
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius)
    return PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )


def unit_sphere_transform(points):
    """
    Returns (center, scale) mapping `points` into a unit sphere at the origin.

    Applying one cloud's transform to another keeps the two comparable: any
    residual offset or scale error between them survives the normalization
    instead of being silently corrected away.
    """
    center = points.reshape(-1, 3).mean(dim=0)
    scale = (points.reshape(-1, 3) - center).norm(dim=1).max()
    return center, scale


def render_points_360(
    points, color=(0.85, 0.45, 0.35), image_size=256, dist=3.0, elev=30.0,
    num_views=36, radius=0.012, device=None,
):
    """
    Orbits a camera around a point cloud of shape (1, N, 3) or (N, 3).

    Mirrors `render_360`, so a point cloud and a mesh rendered with the same
    dist/elev/num_views line up frame for frame.
    """
    if device is None:
        device = get_device()
    points = points.detach()
    if points.dim() == 2:
        points = points.unsqueeze(0)
    points = points.to(device).float()

    rgb = torch.tensor(color, device=device).expand(points.shape).contiguous()
    cloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    renderer = get_points_renderer(
        image_size=image_size, radius=radius, device=device
    )

    frames = []
    for azim in torch.linspace(0, 360, num_views + 1)[:-1]:
        R, T = look_at_view_transform(
            dist=dist, elev=elev, azim=float(azim), device=device
        )
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        frames.append((rend * 255).astype(np.uint8))
    return frames


# ---------------------------------------------------------------------- meshes

def prepare_mesh(mesh, color=(0.85, 0.45, 0.35), center=None, scale=None, device=None):
    """
    Flat-colors a mesh so it can go through `render_360`, optionally normalizing.

    `ico_sphere` and the raw ShapeNet meshes carry no textures, and the Phong
    shader needs some. Passing a shared (center, scale) puts a fit and its target
    in the same frame without correcting away any residual mismatch.
    """
    if device is None:
        device = get_device()
    verts = mesh.verts_list()[0].detach()
    faces = mesh.faces_list()[0].detach()
    if center is not None and scale is not None:
        verts = (verts - center.to(verts.device)) / scale

    textures = TexturesVertex(torch.tensor(color).expand(verts.shape).unsqueeze(0))
    return pytorch3d.structures.Meshes(
        [verts], [faces], textures=textures.to(device)
    ).to(device)
