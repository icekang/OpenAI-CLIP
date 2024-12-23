import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import seaborn as sns
import io

import config as CFG
from modules import ImageEncoder, ProjectionHead
from pathlib import Path
import torchio as tio
from symile import Symile

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        shared_projector=CFG.shared_projector,
        shared_encoder=CFG.shared_encoder,
    ):
        super().__init__()
        self.image_encoder1 = ImageEncoder()
        self.image_projection1 = ProjectionHead(embedding_dim=image_embedding)

        if CFG.shared_encoder:
            self.image_encoder2 = self.image_encoder1
        else:
            self.image_encoder2 = ImageEncoder()
        
        if CFG.shared_projector:
            self.image_projection2 = self.image_projection1
        else:
            self.image_projection2 = ProjectionHead(embedding_dim=image_embedding)

        self.temperature = temperature

    def visualize_clip_loss(self, logits, epoch, suffix="", writer=None):
        import torchvision
        # import matplotlib.pyplot as plt
        import seaborn as sns
        import io
        # import datetime
        # debug_dir = Path("logs" ) /  CFG.experiment_name / "debug_loss"
        # debug_dir.mkdir(exist_ok=True)
        ax = sns.heatmap(logits.detach().cpu().numpy())
        sns_figure = ax.get_figure()
        buf = io.BytesIO()
        sns_figure.savefig(buf, format='png')

        buf.seek(0)
        image = buf.read()
        if writer is not None:
            grid = torchvision.utils.make_grid(image, nrow=1)
            writer.log_image(f"CLIP_Loss/{suffix}", grid, epoch)

        # time_now = str(datetime.datetime.now()).replace(" ", "_")
        # plt.savefig(debug_dir / f'{time_now}_{suffix}.png')
        # plt.close('all')
    
    def visualize_data(self, data, suffix=""):
        import matplotlib.pyplot as plt
        import torchio as tio
        import datetime

        debug_dir = Path("logs" ) /  CFG.experiment_name / "debug_data"
        debug_dir.mkdir(exist_ok=True)

        subjects = dict()
        for batch_idx in range(data["image1"].shape[0]):
            subjects[f'image1_{batch_idx}'] = tio.ScalarImage(tensor=data["image1"][batch_idx].detach().cpu())
            subjects[f'image2_{batch_idx}'] = tio.ScalarImage(tensor=data["image2"][batch_idx].detach().cpu())

        time_now = str(datetime.datetime.now()).replace(" ", "_")

        tio.Subject(
            **subjects
        ).plot(figsize=((batch_idx + 1) * 2, 4), output_path=debug_dir / f"{time_now}_{suffix}_subjects.png")
        plt.close('all')

    def forward(self, batch, mode="train", visualize=False, epoch=None, writer=None):
        image1 = batch["image1"]
        image2 = batch["image2"]

        image1 = torch.nan_to_num(image1, nan=0.0)
        image2 = torch.nan_to_num(image2, nan=0.0)

        # Check for NaN values
        masks = torch.isnan(image1) | torch.isnan(image2)
        masks = ~masks.any(dim=(1, 2, 3, 4))
        image1 = image1[masks]
        image2 = image2[masks]

        # Getting Image and Text Features
        image_features1 = self.image_encoder1(image1)
        image_features2 = self.image_encoder2(image2)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings1 = self.image_projection1(image_features1)
        image_embeddings2 = self.image_projection2(image_features2)
        # assert image_embeddings1.shape == image_embeddings2.shape, f"{image_embeddings1.shape} != {image_embeddings2.shape}"
        # assert len(image_embeddings1.shape) == 2, f"Expected 2D tensor, got {len(image_embeddings1.shape)}D tensor ({image_embeddings1.shape})"

        # image_embeddings1 = F.normalize(image_embeddings1, p=2, dim=1)
        # image_embeddings2 = F.normalize(image_embeddings2, p=2, dim=1)

        # Calculating the Loss
        logits = ((image_embeddings2 @ image_embeddings1.T) + (image_embeddings1 @ image_embeddings2.T)) / 2.0 / self.temperature
        # TODO: Check if logits is NaN or inf or -inf

        # if visualize:
        #     # self.visualize_data(batch, suffix=f"{mode}_data")
        #     self.visualize_clip_loss(F.softmax(logits, dim=-1), suffix=f"{mode}_logits_epoch_{epoch}", writer=writer)
        batch_size = image_embeddings1.shape[0]
        loss = nn.CrossEntropyLoss()(logits, torch.arange(batch_size, device=CFG.device)) # use a simpler loss function 
        # TODO: Check if loss is NaN or inf or -inf
        if torch.isnan(loss).any():
            print("Loss is NaN")
            print("Image Embeddings 1:", image_embeddings1)
            print("Image Embeddings 2:", image_embeddings2)
            print("Logits:", logits)
            print("Cross Entropy Loss:", loss)
            torch.save({
                "image_embeddings1": image_embeddings1,
                "image_embeddings2": image_embeddings2,
                "logits": logits,
                "loss": loss,
                "image1": image1,
                "image2": image2,
                "image_encoder1": self.image_encoder1,
                "image_encoder2": self.image_encoder2,
                "image_projection1": self.image_projection1,
                "image_projection2": self.image_projection
            }, "nan_loss_details.pth")
            # raise ValueError("Loss is NaN")
        return loss.mean(), F.softmax(logits, dim=-1).detach().cpu()

class SymileModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        shared_projector=CFG.shared_projector,
        shared_encoder=CFG.shared_encoder,
    ):
        super().__init__()
        self.image_encoder1 = ImageEncoder()
        self.image_projection1 = ProjectionHead(embedding_dim=image_embedding)

        if CFG.shared_encoder:
            self.image_encoder2 = self.image_encoder1
            self.image_encoder3 = self.image_encoder1
        else:
            self.image_encoder2 = ImageEncoder()
            self.image_encoder3 = ImageEncoder()
        
        if CFG.shared_projector:
            self.image_projection2 = self.image_projection1
            self.image_projection3 = self.image_projection1
        else:
            self.image_projection2 = ProjectionHead(embedding_dim=image_embedding)
            self.image_projection3 = ProjectionHead(embedding_dim=image_embedding)

        self.temperature = temperature
        self.symile_loss = Symile(negative_sampling="n")

    def forward(self, batch, mode="train", visualize=False, epoch=None, writer=None):
        image1 = batch["image1"]
        image2 = batch["image2"]
        image3 = batch["image3"]

        image1 = torch.nan_to_num(image1, nan=0.0)
        image2 = torch.nan_to_num(image2, nan=0.0)
        image3 = torch.nan_to_num(image3, nan=0.0)

        # Check for NaN values
        masks = torch.isnan(image1) | torch.isnan(image2) | torch.isnan(image3)
        masks = ~masks.any(dim=(1, 2, 3, 4))
        image1 = image1[masks]
        image2 = image2[masks]
        image3 = image3[masks]

        # Getting Image and Text Features
        image_features1 = self.image_encoder1(image1)
        image_features2 = self.image_encoder2(image2)
        image_features3 = self.image_encoder3(image3)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings1 = self.image_projection1(image_features1)
        image_embeddings2 = self.image_projection2(image_features2)
        image_embeddings3 = self.image_projection3(image_features3)

        # Calculating the Loss
        loss = self.symile_loss([image_embeddings1, image_embeddings2, image_embeddings3], self.temperature)

        if torch.isnan(loss).any():
            raise ValueError("Loss is NaN")
        return loss.mean(), torch.zeros((1, 1), device='cpu')


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")