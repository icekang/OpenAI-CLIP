import torch

debug = True
captions_path = "."
batch_size = 16
num_workers = 18
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embedding = 1120
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = False # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
IMG_SIZE_W = 512
IMG_SIZE_H = 512
IMG_SIZE_D = 32

CROP_SIZE = 500
CROP_SIZE_D = 32

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1

nnUNet = {
    'dataset_name_or_id': '302',
    'configuration': '3d_32x512x512_b2',
    'fold': 0,
    'trainer_name': 'nnUNetTrainer',
    'plans_identifier': 'nnUNetPlans'
}