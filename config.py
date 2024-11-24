import torch

experiment_name="CLIP_PreIVL_PostStent"
debug = False

visualize_every = 10
model = "clip"
captions_path = "."
skip_data_check = False
batch_size = 16
num_workers = 18
lr = 1e-4
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_norm = 1.0

image_embedding = 1120

pretrained = False # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
IMG_SIZE_W = 512
IMG_SIZE_H = 512
IMG_SIZE_D = 32

CROP_SIZE = 500
CROP_SIZE_D = 32

# CLIP
shared_projector = True
shared_encoder = True

# for projection head; used for both image and text encoders
num_projection_layers = 2
projection_dim = 512
dropout = 0.1

nnUNet = {
    'dataset_name_or_id': '302',
    'configuration': '3d_32x512x512_b2',
    'fold': 0,
    'trainer_name': 'nnUNetTrainer',
    'plans_identifier': 'nnUNetPlans'
}

# nnUNet = {
#     'dataset_name_or_id': '300',
#     'configuration': '3d_fullres',
#     'fold': 'all',
#     'trainer_name': 'nnUNetTrainer',
#     'plans_identifier': 'nnUNetPreprocessPlans',
#     'checkpoint': '/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results/Dataset300_Lumen_and_Wall_OCT/nnUNetTrainer__nnUNetPreprocessPlans__3d_fullres/fold_all/checkpoint_best.pth'
# }