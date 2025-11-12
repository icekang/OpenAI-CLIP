import pandas as pd
from pathlib import Path
import numpy as np

pre_final_coregistration = Path('/nfs/erelab001/shared/Computational_Group/Naravich/datasets/Mariia/Pre_Post')

image_filenames1 = pre_final_coregistration.glob('*Pre_target_Post.nii.gz')
image_filenames2 = pre_final_coregistration.glob('*Post_reg.nii.gz')

image_filenames1 = list(image_filenames1)
image_filenames2 = list(image_filenames2)

image_filenames1.sort()
image_filenames2.sort()

for image_path1, image_path2 in zip(image_filenames1, image_filenames2):
    last_numeric_index = 6
    if not image_path1.name[:last_numeric_index].isnumeric():
        last_numeric_index = 5
    assert image_path1.name[:last_numeric_index] == image_path2.name[:last_numeric_index], f'Image paths do not align, {image_path1} and {image_path2} should have the same case id'

data = pd.DataFrame({
    'image1': image_filenames1,
    'image2': image_filenames2
})
data['id'] = data.index

data.to_csv('pre_post_coregistration.csv', index=False)