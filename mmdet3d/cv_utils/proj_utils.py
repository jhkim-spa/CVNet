import os
import torch
from PIL import Image


@torch.no_grad()
def project_to_image(points, proj_mat):
    device = points.device
    num_pts = points.shape[1]
    points = torch.cat((points, torch.ones((1, num_pts), device=device)))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


@torch.no_grad()
def render_lidar_on_image(points, width, height, lidar2img):
    device = points.device
    if points.shape[1] == 4:
        reflectance = points[:, 3]
    else:
        reflectance = None
    points = points[:, :3]
    proj_velo2cam2 = lidar2img[:3]
    pts_2d = project_to_image(points.transpose(1, 0),
                                    proj_velo2cam2)
    if reflectance is not None:
        shape = (height, width, 5)
    else:
        shape = (height, width, 4)
    pc_projected = torch.zeros(shape, dtype=torch.float32, device=device)
    x_coords = torch.trunc(pts_2d[0]).to(torch.long)
    y_coords = torch.trunc(pts_2d[1]).to(torch.long)
    pc_projected[y_coords, x_coords, :3] = points
    if reflectance is not None:
        pc_projected[y_coords, x_coords, 3] = reflectance
    flag_channel = (pc_projected[:, :, 0] != 0)
    pc_projected[:, :, -1] = flag_channel
    return pc_projected


@torch.no_grad()
def project(points, img_metas):
    """Project point cloud to image"""
    device = points.device
    sample_idx = img_metas['sample_idx']
    pts_filename = img_metas['pts_filename']
    img_filename = os.path.join('/'.join(pts_filename.split('/')[:-2]),
                                'image_2/%0.6d.png' %sample_idx)
    img = Image.open(img_filename)
    width, height = img.size
    lidar2img = torch.tensor(img_metas['lidar2img'], device=device)
    cv = render_lidar_on_image(points, width, height, lidar2img)
    return cv
