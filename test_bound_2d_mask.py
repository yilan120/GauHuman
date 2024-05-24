import numpy as np
from cv2 import Rodrigues as rodrigues
import cv2
import os
import pickle
import struct
import collections
import math
from mano.mano_numpy import MANO
import imageio
from PIL import Image
from torchvision.transforms import ToTensor
import torch
from torchvision.utils import save_image

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Cus_Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Cus_Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras





def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))






def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d



def get_bound_2d_mask(bounds, K, pose, H, W):
    # import ipdb; ipdb.set_trace()
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    # please write code to save mask as an image
    imageio.imwrite('output_image.png', mask)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    imageio.imwrite('output_image1.png', mask)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    imageio.imwrite('output_image2.png', mask)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    imageio.imwrite('output_image3.png', mask)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    imageio.imwrite('output_image4.png', mask)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    imageio.imwrite('output_image5.png', mask)
    return mask


path = '/data/scratch/acw773/HO3D_v2/train/ABF10'

cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


for idx, key in enumerate(cam_extrinsics):

    height = 480
    width = 640

    extr = cam_extrinsics[key]
    intr = cam_intrinsics[extr.camera_id]

    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    focal_length_x = intr.params[0]
    focal_length_y = intr.params[1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)

    w2c = np.eye(4)
    w2c[:3,:3] = R
    w2c[:3,3:4] = T.reshape(3, 1)
    # print("w2c[:3]:{}".format(w2c[:3]))
    K = np.array([[focal_length_x, 0, intr.params[2]], [0, focal_length_x, intr.params[3]], [0, 0, 1]])

    
    image_path = os.path.join(os.path.join(path, 'images'), os.path.basename(extr.name))
    image = Image.open(image_path)
    image = ToTensor()(image)
    meta_pth = image_path.replace("images", "meta").replace(".png", ".pkl")
    print(image_path)
    with open(meta_pth, 'rb') as f:
        meta_data = pickle.load(f)
    handpose = meta_data['handPose'].astype('float32')
    handshape = meta_data['handBeta'].astype('float32')
    mano_model = MANO(left='right', model_dir='assets')
    xyz, _ = mano_model(handpose, handshape)

    # obtain the original bounds for point sampling
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    world_bound = np.stack([min_xyz, max_xyz], axis=0)

    print("world_bound:{}".format(world_bound))

    bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], height, width)
    bound_mask = Image.fromarray(np.array(bound_mask, dtype=np.byte))
    # bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))
    bound_mask.save("/data/home/acw773/GauHuman/img1.png","PNG")

    mask_array = np.array(bound_mask)
    print("Unique values in the mask:", np.unique(mask_array))

    num_pixels_value_one = np.sum(mask_array == 1)

    print("Number of pixels with value = 1:", num_pixels_value_one)

    # mask = torch.tensor(bound_mask).unsqueeze(0)/255.
    # print('mask shape:{}'.format(mask.shape))
    # print('image shape:{}'.format(image.shape))
    # print(mask.sum())
    # sub_image = image * mask
    # save_image(image, 'image.png')
    # save_image(sub_image, 'sub_image.png')


    break