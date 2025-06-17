import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
import math

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        first_frame: int = 0
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, f"colmap_{first_frame}", "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse/0/")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        camera_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(camera_names)
        camera_names = [camera_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # # Load images.
        # if factor > 1 and not self.extconf["no_factor_suffix"]:
        #     image_dir_suffix = f"_{factor}"
        # else:
        #     image_dir_suffix = ""
        # colmap_image_dir = os.path.join(data_dir, f"colmap_{first_frame}", "images")
        # image_dir = os.path.join(data_dir, f"colmap_{first_frame}", "images" + image_dir_suffix)
        # for d in [image_dir, colmap_image_dir]:
        #     if not os.path.exists(d):
        #         raise ValueError(f"Image folder {d} does not exist.")

        # # Downsampled images may have different names vs images used for COLMAP,
        # # so we need to map between the two sorted lists of files.
        # colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        # image_files = sorted(_get_rel_paths(image_dir))
        # colmap_to_image = dict(zip(colmap_files, image_files))
        # image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.camera_names = camera_names  # List[str], (num_images,)
        # self.image_paths = image_paths  # List[str], (num_images,)
        self.campaths = [os.path.join(data_dir, "png", x) for x in sorted(os.listdir(os.path.join(data_dir, "png")))]
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(os.path.join(self.campaths[0],f"{(first_frame+1):05d}.png"))[..., :3]
        actual_height, actual_width = [x // factor for x in actual_image.shape[:2]]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""
    # | 参数名           | 作用                                |
    # | ------------- | --------------------------------- |
    # | `parser`      | 上面定义的 `Parser` 类，包含所有 COLMAP 处理结果 |
    # | `split`       | `"train"` 或 `"test"`              |
    # | `patch_size`  | 如果指定，则会对图像做随机裁剪                   |
    # | `load_depths` | 是否加载深度投影（可选）                      |
    # | `start_frame` | 当前 GOP 的起始帧号                      |
    # | `GOP_size`    | 当前 GOP 的帧数                        |
    # | `test_set`    | 指定测试的 GOP 编号（如 `[0]` 表示 GOP\_0）   |
    # | `remove_set`  | 要从训练集中排除的 GOP 编号                  |

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        start_frame: int = 0,
        GOP_size: int = 50,
        test_set: list = [0],
        remove_set: list = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.start_frame = start_frame
        self.GOP_size = GOP_size
        indices = np.arange(len(self.parser.camera_names)*GOP_size)
        self.cameras_length = len(self.parser.camera_names)
        if split == "train":
            self.indices = [x for x in indices if (x // GOP_size) not in test_set]
        else:
            self.indices = [x for x in indices if (x // GOP_size) in test_set]

        if remove_set is not None:
            self.indices = [x for x in self.indices if (x // GOP_size) not in remove_set]
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        cam = self.indices[item] // self.GOP_size
        frame_idx = self.start_frame + item - (item // self.GOP_size) * self.GOP_size
        camera_path = self.parser.campaths[cam]
        image_path = os.path.join(camera_path, f"{(frame_idx+1):05d}.png")
        image = imageio.imread(image_path)[..., :3]
        image = cv2.resize(image, dsize=(image.shape[1]//self.parser.factor, image.shape[0]//self.parser.factor), interpolation=cv2.INTER_LINEAR)
        camera_id = self.parser.camera_ids[cam]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[cam]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y


# 每个GOP中所有图片相同的内容
#         | 字段           | 说明                                  |
# | ------------ | ----------------------------------- |
# | `K`          | 相机内参矩阵，所有帧来自同一个摄像头                  |
# | `camera_id`  | 恒定为 camera\_0 对应的 ID                |
# | `camtoworld` | 相机位姿矩阵（整个 GOP 用的是同一帧的，通常是第一帧）       |
# | `mask`       | 如果存在 mask（如 fisheye），所有帧用的是同一个 mask |
# 不同的内容：
# | 字段         | 说明                                     |
# | ---------- | -------------------------------------- |
# | `image`    | 当前帧图像（如 cam00 的第 i 帧图像）                |
# | `image_id` | 样本在整个 `Dataset` 中的索引（0 \~ 239）         |
# | `time`     | 归一化时间戳：`i / (GOP_size - 1)`，范围 \[0, 1] |
        # {
        #     "K": 相机内参 (torch.Tensor, [3, 3]),
        #     "camtoworld": 外参矩阵 (torch.Tensor, [4, 4]),
        #     "image": 图像像素值 (torch.Tensor, [H, W, 3]),
        #     "image_id": 当前帧索引,
        #     "time": 当前帧在 GOP 中的归一化时间, 例如 0.0 ~ 1.0,
        #     "camera_id": 当前相机索引
        # }
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "time": float(frame_idx - self.start_frame) / (self.GOP_size-1),
            "camera_id": camera_id - 1 if self.split == "train" else -1,
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # 点云深度信息部分是注释掉的
        # if self.load_depths:
        #     # projected points to image plane to get depths
        #     worldtocams = np.linalg.inv(camtoworlds)
        #     image_name = self.parser.image_names[cam]
        #     point_indices = self.parser.point_indices[image_name]
        #     points_world = self.parser.points[point_indices]
        #     points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
        #     points_proj = (K @ points_cam.T).T
        #     points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
        #     depths = points_cam[:, 2]  # (M,)
        #     # filter out points outside the image
        #     selector = (
        #         (points[:, 0] >= 0)
        #         & (points[:, 0] < image.shape[1])
        #         & (points[:, 1] >= 0)
        #         & (points[:, 1] < image.shape[0])
        #         & (depths > 0)
        #     )
        #     points = points[selector]
        #     depths = depths[selector]
        #     data["points"] = torch.from_numpy(points).float()
        #     data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="test", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
