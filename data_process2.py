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

# Path to the directory with all components
dataset_dir = './data2/'

from os import listdir
from os.path import isfile, join

dir = './data2/camera_image'
files = [f for f in listdir(dir) if isfile(join(dir, f))]
#print(len(files))
files = [f.split('.')[0] for f in files if len(f.split('.')) == 2]
files = [sys.argv[1].split('/')[-1].split('.')[0]]
#print(files)

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
  cam_box_df = read('camera_box', context_name)
  
  # print(cam_image_df.head())
  # print(cam_box_df.head())
  df_temp = cam_box_df[cam_box_df['key.camera_name'] == 5]
  cam_image_df = v2.merge(cam_image_df, cam_box_df, right_group=True)
  # print(cam_image_df.head())
  del cam_box_df
  gc.collect()
  
  # Example how to access data fields via v2 object-oriented API
  print(f'Available {cam_image_df.shape[0]} rows:')
  for i, (_, r) in enumerate(cam_image_df.iterrows()):
    # Create component dataclasses for the raw data
    cam_image = v2.CameraImageComponent.from_dict(r)
    cam_nm = cam_image.key.camera_name
    #print(cam_nm)
    cam_box = v2.CameraBoxComponent.from_dict(r)

    img = tf.io.decode_jpeg(cam_image.image).numpy()
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    boxes = []
    #print(zip(cam_box.box.center.x, cam_box.box.center.y, cam_box.box.size.x, cam_box.box.size.y))
    for j, (x, y, w, h, t) in enumerate(zip(cam_box.box.center.x, cam_box.box.center.y, cam_box.box.size.x, cam_box.box.size.y, cam_box.type)):
        x /= 4
        y /= 4
        w /= 4
        h /= 4
        x1 = x-w/2
        x2 = x+w/2
        y1 = y-h/2
        y2 = y+h/2
        boxes.append([x1, y1, x2, y2, t])
        #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    boxes = np.array(boxes)
    

    filename = path+str(cam_image.key.frame_timestamp_micros)+'_boxes_'+str(cam_nm)+".npy"
    np.save(filename, boxes)



  del cam_image_df
  gc.collect()