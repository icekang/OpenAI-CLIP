import os
import cv2
import torch
import SimpleITK as sitk
import numpy as np
from torchvision import transforms
import torchio as tio
import config as CFG


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames1, image_filenames2, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames1 = image_filenames1
        self.image_filenames1.sort()
        self.image_filenames2 = image_filenames2
        self.image_filenames2.sort()

        for image_path1, image_path2 in zip(self.image_filenames1, self.image_filenames2):
            assert str(image_path1)[:6] == str(image_path2)[:6], f'Image paths do not align, {image_path1} and {image_path2} should have the same case id'

        self.transforms = transforms

    def read_image(self, image_filename):
        itk_image = sitk.ReadImage(image_filename)
        npy_image = sitk.GetArrayFromImage(itk_image)
        if npy_image.ndim == 3:
            # 3d, as in original nnunet
            npy_image = npy_image[None]
        elif npy_image.ndim == 4:
            # 4d, multiple modalities in one file
            pass
        else:
            raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {image_filename}")

        return npy_image.astype(np.float32)
    
    def __getitem__(self, idx, z_index=None):
        item = dict()

        image1 = self.read_image(self.image_filenames1[idx])
        image2 = self.read_image(self.image_filenames2[idx])

        max_depth = min(image1.shape[1], image2.shape[1])
        
        if z_index is None:
            while True:
                z_index = np.random.randint(0, max_depth - CFG.CROP_SIZE_D)
                std1 = image1[:, z_index:z_index + CFG.CROP_SIZE_D, ...].std()
                std2 = image2[:, z_index:z_index + CFG.CROP_SIZE_D, ...].std()
                if std1 > 0 and std2 > 0:
                    break

        image1 = image1[:, z_index:z_index + CFG.CROP_SIZE_D, ...]
        image2 = image2[:, z_index:z_index + CFG.CROP_SIZE_D, ...]

        image1 = self.transforms(image1.transpose(0, 2, 3, 1))
        image2 = self.transforms(image2.transpose(0, 2, 3, 1))

        item['image1'] = torch.tensor(image1.transpose(0, 3, 1, 2)).float()
        item['image2'] = torch.tensor(image2.transpose(0, 3, 1, 2)).float()

        item['image_path1'] = self.image_filenames1[idx]
        item['image_path2'] = self.image_filenames2[idx]

        item['image_z_index'] = z_index

        return item

    def __len__(self):
        return len(self.image_filenames1)



def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose(
            [
                tio.transforms.CropOrPad((CFG.CROP_SIZE, CFG.CROP_SIZE, CFG.CROP_SIZE_D)),
                tio.transforms.Resize((CFG.IMG_SIZE_W, CFG.IMG_SIZE_H, CFG.IMG_SIZE_D)),
                tio.transforms.ZNormalization()
            ]
        )
    else:
        return transforms.Compose(
            [
                tio.transforms.CropOrPad((CFG.CROP_SIZE, CFG.CROP_SIZE, CFG.CROP_SIZE_D)),
                tio.transforms.Resize((CFG.IMG_SIZE_W, CFG.IMG_SIZE_H, CFG.IMG_SIZE_D)),
                tio.transforms.ZNormalization()
            ]
        )

def get_optimized_dataloaders(image_filenames1, image_filenames2, transforms):
    subjects = []
    for image_path1, image_path2 in zip(image_filenames1, image_filenames2):
        shape1 = sitk.ReadImage(image_path1)
        shape1 = shape1.GetSize()
        direction1 = sitk.ReadImage(image_path1).GetDirection()
        shape2 = sitk.ReadImage(image_path2)
        shape2 = shape2.GetSize()
        direction2 = sitk.ReadImage(image_path2).GetDirection()
        if shape1 != shape2 or direction1 != direction2:
            if shape1[-1] < shape2[-1]:
                larger_image_path = image_path2
                smaller_image_path = image_path1
                smaller_shape = shape1
            else:
                larger_image_path = image_path1
                smaller_image_path = image_path2
                smaller_shape = shape2

            larger_image = sitk.GetArrayFromImage(sitk.ReadImage(larger_image_path))
            larger_image = larger_image [:smaller_shape[-1], ...]
            larger_image = sitk.GetImageFromArray(larger_image)
            larger_image.SetSpacing(sitk.ReadImage(smaller_image_path).GetSpacing())
            larger_image.SetDirection(sitk.ReadImage(smaller_image_path).GetDirection())

            import shutil
            shutil.move(larger_image_path, larger_image_path.replace('.nii.gz', '_original.nii.gz'))
            print(f"Resizing {larger_image_path} to {smaller_shape} (Original image has been saved as {larger_image_path.replace('.nii.gz', '_original.nii.gz')})")
            sitk.WriteImage(larger_image, larger_image_path)

    for image_path1, image_path2 in zip(image_filenames1, image_filenames2):
        subject = tio.Subject(
            image1=tio.ScalarImage(image_path1),
            image2=tio.ScalarImage(image_path2),
            non_zero_mask1=tio.ScalarImage(image_path1),
        )
        subjects.append(subject)
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transforms)
    patches_queue = tio.Queue(
        subjects_dataset,
        max_length=400,
        samples_per_volume=10,
        sampler=tio.data.WeightedSampler(patch_size=[512, 512, 32], probability_map="non_zero_mask1"),
        num_workers=CFG.num_workers,
        shuffle_subjects=True,
    )
    patches_loader = torch.utils.data.DataLoader(
        patches_queue,
        batch_size=CFG.batch_size,
        num_workers=0,  # this must be 0
    )
    return patches_loader

def get_torchio_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose(
            [
                tio.transforms.CropOrPad((CFG.CROP_SIZE, CFG.CROP_SIZE, 375)),
                tio.transforms.Resize((CFG.IMG_SIZE_W, CFG.IMG_SIZE_H, 375)),
                tio.transforms.ZNormalization(exclude=["non_zero_mask1"])
            ]
        )
    else:
        return transforms.Compose(
            [
                tio.transforms.CropOrPad((CFG.CROP_SIZE, CFG.CROP_SIZE, 375)),
                tio.transforms.Resize((CFG.IMG_SIZE_W, CFG.IMG_SIZE_H, 375)),
                tio.transforms.ZNormalization(exclude=["non_zero_mask1"])
            ]
        )