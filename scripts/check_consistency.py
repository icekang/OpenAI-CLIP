import SimpleITK as sitk
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

# Grab the arguments that are passed in
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

dataframe = pd.read_csv('pre_post_coregistration_corrected.csv')
image_filenames1=dataframe["image1"].values[my_task_id:len(dataframe):num_tasks]
image_filenames2=dataframe["image2"].values[my_task_id:len(dataframe):num_tasks]
image_filenames3=dataframe["image3"].values[my_task_id:len(dataframe):num_tasks]

for image_path1, image_path2, image_path3 in zip(image_filenames1, image_filenames2, image_filenames3):
    image1 = sitk.ReadImage(image_path1)
    shape1 = image1.GetSize()
    direction1 = image1.GetDirection()
    image2 = sitk.ReadImage(image_path2)
    shape2 = image2.GetSize()
    direction2 = image2.GetDirection()
    image3 = sitk.ReadImage(image_path3)
    shape3 = image3.GetSize()
    direction3 = image3.GetDirection()
    
    assert direction1 == direction2 == direction3, f"Direction mismatch: {direction1}, {direction2}, {direction3} ({image_path1}, {image_path2}, {image_path3})"
    assert shape1 == shape2 == shape3, f"Shape mismatch: {shape1}, {shape2}, {shape3} ({image_path1}, {image_path2}, {image_path3})"
