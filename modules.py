import torch
from torch import nn
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
import copy
import config as CFG


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, trainable=CFG.trainable
    ):
        super().__init__()
        trainer = get_trainer_from_args(
            dataset_name_or_id=CFG.nnUNet['dataset_name_or_id'],
            configuration=CFG.nnUNet['configuration'],
            fold=CFG.nnUNet['fold'],
            trainer_name=CFG.nnUNet['trainer_name'],
            plans_identifier=CFG.nnUNet['plans_identifier'],
            device=torch.device('cpu'))
        trainer.initialize()
        # trainer.load_checkpoint(CFG.nnUNet['checkpoint']) # load the checkpoint if needed
        self.model = copy.deepcopy(trainer.network.encoder) # only the encoder part will be used for nnUNet
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        del trainer

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        features = self.model(x)
        features = [self.pool(feature) for feature in features]
        features = torch.cat(features, dim=1)
        return features


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

