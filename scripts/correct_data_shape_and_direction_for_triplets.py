import sys
import SimpleITK as sitk
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np


# Grab the arguments that are passed in
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

dataframe = pd.read_csv('/home/gridsan/nchutisilp/projects/OpenAI-CLIP/pre_post_final_coregistration.csv')

image_filenames1=dataframe["image1"].values[my_task_id:len(dataframe):num_tasks]
image_filenames2=dataframe["image2"].values[my_task_id:len(dataframe):num_tasks]
image_filenames3=dataframe["image3"].values[my_task_id:len(dataframe):num_tasks]

output_dir = Path('/home/gridsan/nchutisilp/datasets/SymileMariia_PrePostFinal')
output_dir.mkdir(exist_ok=True, parents=True)

corrected_image_filenames = [[], [], []] # [image_filenames1, image_filenames2, image_filenames3]

for image_path1, image_path2, image_path3 in tqdm(zip(image_filenames1, image_filenames2, image_filenames3), total=len(image_filenames1)):
    image1 = sitk.ReadImage(image_path1)
    shape1 = image1.GetSize()
    direction1 = image1.GetDirection()
    image2 = sitk.ReadImage(image_path2)
    shape2 = image2.GetSize()
    direction2 = image2.GetDirection()
    image3 = sitk.ReadImage(image_path3)
    shape3 = image3.GetSize()
    direction3 = image3.GetDirection()

    is_shape_mismatched = shape1 != shape2 or shape1 != shape3 or shape2 != shape3
    is_direction_mismatched = direction1 != direction2 or direction1 != direction3 or direction2 != direction3
    if is_shape_mismatched or is_direction_mismatched:
        smaller_shape_index = np.argmin([shape1[-1], shape2[-1], shape3[-1]])
        smaller_shape = [shape1, shape2, shape3][smaller_shape_index]
        smaller_image_path = [image_path1, image_path2, image_path3][smaller_shape_index]
        smaller_image = sitk.ReadImage(smaller_image_path)

        for i in range(3):
            if i != smaller_shape_index:
                larger_image_path = [image_path1, image_path2, image_path3][i]
                larger_image = sitk.GetArrayFromImage(sitk.ReadImage(larger_image_path))
                larger_image = larger_image[:smaller_shape[-1], ...]
                larger_image = sitk.GetImageFromArray(larger_image)
                larger_image.SetSpacing(smaller_image.GetSpacing())
                larger_image.SetDirection(smaller_image.GetDirection())
                sitk.WriteImage(larger_image, output_dir / Path(larger_image_path).name)
            else:
                image_path = [image_path1, image_path2, image_path3][i]
                # shutil.copy(image_path, output_dir / Path(image_path).name)
                sitk.WriteImage(smaller_image, output_dir / Path(image_path).name)
