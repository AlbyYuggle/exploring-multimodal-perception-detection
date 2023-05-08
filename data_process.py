import tensorflow as tf
import sys
import time
def _combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.
  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.
  Args:
    tensor: A tensor of any type.
  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(input=tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def scatter_nd_with_pool(index,
                         value,
                         shape,
                         pool_method=tf.math.unsorted_segment_max):
  """Similar as tf.scatter_nd but allows custom pool method.
  tf.scatter_nd accumulates (sums) values if there are duplicate indices.
  Args:
    index: [N, 2] tensor. Inner dims are coordinates along height (row) and then
      width (col).
    value: [N, ...] tensor. Values to be scattered.
    shape: (height,width) list that specifies the shape of the output tensor.
    pool_method: pool method when there are multiple points scattered to one
      location.
  Returns:
    image: tensor of shape with value scattered. Missing pixels are set to 0.
  """
  if len(shape) != 2:
    raise ValueError('shape must be of size 2')
  height = shape[0]
  width = shape[1]
  # idx: [N]
  index_encoded, idx = tf.unique(index[:, 0] * width + index[:, 1])
  value_pooled = pool_method(value, idx, tf.size(input=index_encoded))
  index_unique = tf.stack(
      [index_encoded // width,
       tf.math.mod(index_encoded, width)], axis=-1)
  shape = [height, width]
  value_shape = _combined_static_and_dynamic_shape(value)
  if len(value_shape) > 1:
    shape = shape + value_shape[1:]

  image = tf.scatter_nd(index_unique, value_pooled, shape)
  return image

def build_camera_depth_image(range_image_cartesian,
                             extrinsic,
                             camera_projection,
                             camera_image_size,
                             camera_name,
                             pool_method=tf.math.unsorted_segment_min,
                             scope=None):
  """Builds camera depth image given camera projections.
  The depth value is the distance between a lidar point and camera frame origin.
  It is decided by cartesian coordinates in vehicle frame and the camera
  extrinsic. Optionally, the cartesian coordinates can be set in the vehicle
  frame corresponding to each pixel pose which makes the depth generated to have
  vehicle motion taken into account.
  Args:
    range_image_cartesian: [B, H, W, 3] tensor. Range image points in vehicle
      frame. Note that if the range image is provided by pixel_pose, then you
      can optionally pass in the cartesian coordinates in each pixel frame.
    extrinsic: [B, 4, 4] tensor. Camera extrinsic.
    camera_projection: [B, H, W, 6] tensor. Each range image pixel is associated
      with at most two camera projections. See dataset.proto for more details.
    camera_image_size: a list of [height, width] integers.
    camera_name: an integer that identifies a camera. See dataset.proto.
    pool_method: pooling method when multiple lidar points are projected to one
      image pixel.
    scope: the name scope.
  Returns:
    image: [B, height, width] depth image generated.
  """
  with tf.compat.v1.name_scope(
      scope, 'BuildCameraDepthImage',
      [range_image_cartesian, extrinsic, camera_projection]):
    # [B, 4, 4]
    vehicle_to_camera = tf.linalg.inv(extrinsic)
    # [B, 3, 3]
    vehicle_to_camera_rotation = vehicle_to_camera[:, 0:3, 0:3]
    # [B, 3]
    vehicle_to_camera_translation = vehicle_to_camera[:, 0:3, 3]
    # [B, H, W, 3]
    range_image_camera = tf.einsum(
        'bij,bhwj->bhwi', vehicle_to_camera_rotation,
        range_image_cartesian) + vehicle_to_camera_translation[:, tf.newaxis,
                                                               tf.newaxis, :]
    # [B, H, W]
    range_image_camera_norm = tf.norm(tensor=range_image_camera, axis=-1)
    camera_projection_mask_1 = tf.tile(
        tf.equal(camera_projection[..., 0:1], camera_name), [1, 1, 1, 2])
    camera_projection_mask_2 = tf.tile(
        tf.equal(camera_projection[..., 3:4], camera_name), [1, 1, 1, 2])
    camera_projection_selected = tf.ones_like(
        camera_projection[..., 1:3], dtype=camera_projection.dtype) * -1
    camera_projection_selected = tf.compat.v1.where(camera_projection_mask_2,
                                                    camera_projection[..., 4:6],
                                                    camera_projection_selected)
    # [B, H, W, 2]
    camera_projection_selected = tf.compat.v1.where(camera_projection_mask_1,
                                                    camera_projection[..., 1:3],
                                                    camera_projection_selected)
    # [B, H, W]
    camera_projection_mask = tf.logical_or(camera_projection_mask_1,
                                           camera_projection_mask_2)[..., 0]
    # np.set_printoptions(threshold=sys.maxsize)
    # # print(camera_projection_mask.numpy())
    # print(camera_projection_selected.numpy())
    def fn(args):
      """Builds depth image for a single frame."""

      # NOTE: Do not use ri_range > 0 as mask as missing range image pixels are
      # not necessarily populated as range = 0.
      mask, ri_range, cp = args
      mask_ids = tf.compat.v1.where(mask)
      index = tf.gather_nd(
          tf.stack([cp[..., 1], cp[..., 0]], axis=-1), mask_ids)
      value = tf.gather_nd(ri_range, mask_ids)
      #return tf.scatter_nd(index, value, camera_image_size)
      return scatter_nd_with_pool(index, value, camera_image_size, pool_method)

    images =  tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems=[
            camera_projection_mask, range_image_camera_norm,
            tf.cast(camera_projection_selected, dtype=tf.int64)
        ],
        dtype=range_image_camera_norm.dtype))
    return images

#@title Initial setup
from typing import Optional
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
import matplotlib.pyplot as plt
from PIL import Image
import os
from waymo_open_dataset.utils import range_image_utils
import torchvision
import gc
import numpy as np
import sys
#import open3d as o3d
import cv2
import time
import psutil

# Path to the directory with all components
dataset_dir = './data2/'

from os import listdir
from os.path import isfile, join

dir = './data2/camera_image'
files = [f for f in listdir(dir) if isfile(join(dir, f))]
#print(len(files))
files = [f.split('.')[0] for f in files if len(f.split('.')) == 2]
#print(files)
files = [sys.argv[1].split('/')[-1].split('.')[0]]
print(files)
def read(tag: str, context_name: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  pathname = f'{dataset_dir}/{tag}/{context_name}.parquet'
  return pd.read_parquet(pathname, engine='pyarrow')
  return dd.read_parquet(paths)

for context_name in files:
  print(context_name)
  # @title Basic Example (Camera images with labels)
  path = './data/' + context_name+'/'
  isExist = os.path.exists(path)
  if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path)
  # Lazily read camera images and boxes 
  cam_image_df = read('camera_image', context_name)
  #cam_box_df = read('camera_box', context_name)
  #lidar_box_df = read('projected_lidar_box', context_name)
  lidar_df = read('lidar', context_name)
  lidar_cam_proj_df = read('lidar_camera_projection', context_name)
  cam_calib_df = read('camera_calibration', context_name)
  lidar_calib_df = read('lidar_calibration', context_name)
  #cam_lidar_assoc_df = read('camera_to_lidar_box_association', context_name)
  # Combine DataFrame for individual components into a single DataFrame.
  # Camera cam_box_df will be grouped, so each row will have a camera image
  # and all associated boxes.
  # cam_image_w_box_df = v2.merge(cam_box_df, cam_image_df)
  # cam_image_w_box_df_calib = v2.merge(cam_image_w_box_df, cam_calib_df)

  # cam_obj_df = v2.merge(cam_lidar_assoc_df, cam_image_w_box_df)
  # obj_df = v2.merge(cam_obj_df, lidar_box_df)
  # lidar_proj_df = v2.merge(lidar_df, lidar_cam_proj_df)
  # df = v2.merge(lidar_proj_df, obj_df, left_group=True, right_group=True)

  lidar_df = lidar_df.merge(lidar_cam_proj_df)
  lidar_df = lidar_df.merge(lidar_calib_df)

  cam_image_df = cam_image_df.merge(cam_calib_df)

  cam_image_df = cam_image_df.merge(lidar_df)
  del lidar_df
  del lidar_cam_proj_df
  del cam_calib_df
  del lidar_calib_df
  gc.collect()
  # Example how to access data fields via v2 object-oriented API
  #print(f'Available {df.shape[0].compute()} rows:')
  for i, (_, r) in enumerate(cam_image_df.iterrows()):
    # Create component dataclasses for the raw data
    cam_image = v2.CameraImageComponent.from_dict(r)
    lidar = v2.LiDARComponent.from_dict(r)
    lidar_cam_proj = v2.LiDARCameraProjectionComponent.from_dict(r)
    cam_calib = v2.CameraCalibrationComponent.from_dict(r)
    lidar_calib = v2.LiDARCalibrationComponent.from_dict(r)

    lidar_cart = lidar.range_image_return1.values.reshape(lidar.range_image_return1.shape)[:, :, :3]
    lidar_cart = np.expand_dims(lidar_cart, axis=0)
    lidar_cart = tf.convert_to_tensor(lidar_cart, dtype=tf.float64)
    #print(lidar.range_image_return1.shape)

    #print(cam_calib.extrinsic.transform)
    ext_mat = np.reshape(cam_calib.extrinsic.transform, (1, 4, 4))
    ext_mat = tf.convert_to_tensor(ext_mat, dtype=tf.float64)
    #print(ext_mat)

    lidar_ext = np.reshape(lidar_calib.extrinsic.transform, (1, 4, 4))
    lidar_ext = tf.convert_to_tensor(lidar_ext, dtype=tf.float64)

    cam_proj = lidar_cam_proj.range_image_return1.values.reshape(lidar_cam_proj.range_image_return1.shape)
    cam_proj = np.expand_dims(cam_proj, axis=0)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(cam_proj)
    cam_proj = tf.convert_to_tensor(cam_proj, dtype=tf.float64)
    cam_img_sz = [cam_calib.height, cam_calib.width]
    cam_nm = cam_image.key.camera_name
    if(lidar_calib.beam_inclination.values is None):
      #print('None')
      continue
    lidar_inc = tf.expand_dims(tf.convert_to_tensor(lidar_calib.beam_inclination.values, dtype=tf.float64), axis=0)
    #print(lidar_inc.shape, tf.expand_dims(lidar_cart[0, :,:,0], axis=1).shape)
    point_cloud = range_image_utils.extract_point_cloud_from_range_image(tf.expand_dims(lidar_cart[0, :,:,0], axis=0), lidar_ext, lidar_inc, dtype=tf.float64)

    img = tf.io.decode_jpeg(cam_image.image).numpy()
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    filename = path+str(cam_image.key.frame_timestamp_micros)+'_cam_'+str(cam_nm)+".jpg"
    cv2.imwrite(filename, img)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud.numpy().reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd])

    # np.set_printoptions(threshold=sys.maxsize)
    # print(lidar.range_image_return1.shape)
    # print(lidar.range_image_return1.values.reshape(lidar.range_image_return1.shape)[:, :, :3])

    # print(lidar.range_image_return1.shape)
    # print()
    # print(tf.io.decode_jpeg(cam_image.image).shape)
    # print(lidar_cam_proj.range_image_return1.shape)
    # print(type(lidar_cam_proj.range_image_return1.values))
    # print(cam_calib.extrinsic.transform)
    # print(type(cam_calib.extrinsic.transform))
    # print(cam_calib.width, cam_calib.height)
    # print(cam_image.key.camera_name)

    #print(point_cloud.shape, cam_proj.shape)
    rgbd_image = build_camera_depth_image(point_cloud, ext_mat, cam_proj, cam_img_sz, cam_nm)
    
    
    dim = rgbd_image.numpy()[0]

    filename = path+str(cam_image.key.frame_timestamp_micros)+'_depth_'+str(cam_nm)+".jpg"
    cv2.imwrite(filename, dim)
    # dim[dim>0] = 255
    # dim = dim.astype(np.int64)
    # #np.set_printoptions(threshold=sys.maxsize)
    # plt.imshow(dim, cmap='gray')
    # plt.show()
  del cam_image_df
  gc.collect()
  


      