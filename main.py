import os
import gc
import numpy as np
import pandas as pd
import torch.amp
from tqdm import tqdm

import torch
from torch import nn

import config as CFG
from dataset import CLIPDataset, get_transforms, get_optimized_dataloaders, get_torchio_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr
import torchio as tio
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from argparse import ArgumentParser

def make_logdir():
    logdir = Path("logs") / CFG.experiment_name
    logdir.mkdir(exist_ok=True, parents=True)

    debug_data = logdir / "debug_data"
    debug_data.mkdir(exist_ok=True)

    debug_loss = logdir / "debug_loss"
    debug_loss.mkdir(exist_ok=True)

    return logdir

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/pre_final_coregistration.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, mode):
    # transforms = get_transforms(mode=mode)
    transforms = get_torchio_transforms(mode=mode)
    dataloader = get_optimized_dataloaders(
        image_filenames1=dataframe["image1"].values,
        image_filenames2=dataframe["image2"].values,
        transforms=transforms,
    )
    # dataset = CLIPDataset(
    #     dataframe["image1"].values,
    #     dataframe["image2"].values,
    #     transforms=transforms,
    # )
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=CFG.batch_size,
    #     num_workers=CFG.num_workers,
    #     shuffle=True if mode == "train" else False,
    # )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, scaler, epoch=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    optimizer.zero_grad(set_to_none=True)

    softmax_logits = []
    for iteration, batch in enumerate(tqdm_object):
        device = CFG.device
        batch['image1'] = batch['image1'][tio.DATA].permute(0, 1, 4, 3, 2).to(device)
        batch['image2'] = batch['image2'][tio.DATA].permute(0, 1, 4, 3, 2).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            visualize = iteration % CFG.visualize_every == 0
            loss, softmax_logit = model(batch, visualize=visualize, epoch=epoch)
            if loss.isnan():
                print("Loss is NaN, skip updating model")
                continue
            softmax_logits.append(softmax_logit)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == "batch":
            lr_scheduler.step(loss)

        count = batch["image1"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter, softmax_logits


def valid_epoch(model, valid_loader, epoch=None):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    softmax_logits = []
    for iteration, batch in enumerate(tqdm_object):
        device = CFG.device
        batch['image1'] = batch['image1'][tio.DATA].permute(0, 1, 4, 3, 2).to(device)
        batch['image2'] = batch['image2'][tio.DATA].permute(0, 1, 4, 3, 2).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # batch = {k: v.to(CFG.device) for k, v in batch.items() if k not in ["caption", "image_path1", "image_path2", "image_z_index"]}

            visualize = iteration % CFG.visualize_every == 0
            loss, softmax_logit = model(batch, mode="valid", visualize=visualize, epoch=epoch)
            softmax_logits.append(softmax_logit)

        count = batch["image1"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter, softmax_logits


def visualize_clip_loss(softmax_logits, suffix, writer, epoch):
    import torchvision
    import seaborn as sns
    import io
    from PIL import Image
    import matplotlib.pyplot as plt
    images = []
    for logits in softmax_logits:
        ax = sns.heatmap(logits.detach().cpu().numpy())
        sns_figure = ax.get_figure()
        buf = io.BytesIO()
        sns_figure.savefig(buf, format='png')

        buf.seek(0)
        image = buf.read()
        image = Image.open(io.BytesIO(image))
        image = torchvision.transforms.functional.pil_to_tensor(image)
        images.append(image)
        plt.clf()
    grid = torchvision.utils.make_grid(images, nrow=1)
    writer.add_image(f"CLIP_Loss/{suffix}", grid, epoch)

def main():
    writer = SummaryWriter(Path("logs") / CFG.experiment_name / "runs")
    train_df, valid_df = make_train_valid_dfs()
    train_loader = build_loaders(train_df, mode="train")
    valid_loader = build_loaders(valid_df, mode="valid")


    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, softmax_logits = train_epoch(model, train_loader, optimizer, lr_scheduler, step, scaler, epoch=epoch)
        if step == "epoch":
            lr_scheduler.step(train_loss.avg)
        writer.add_scalar("Loss/train", train_loss.avg, epoch)
        visualize_clip_loss(softmax_logits, "train", writer, epoch)
        model.eval()
        with torch.no_grad():
            valid_loss, softmax_logits = valid_epoch(model, valid_loader)
            writer.add_scalar("Loss/valid", valid_loss.avg, epoch)
            visualize_clip_loss(softmax_logits, "valid", writer, epoch)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), Path("logs") / CFG.experiment_name / "best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    if args.config is not None:
        import yaml
        with open(args.config, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        for key, value in config.items():
            setattr(CFG, key, value)
    print("Start training with the following configuration:")
    print(CFG)

    print("Setting seed to be:", args.seed)
    seed_everything(args.seed)

    make_logdir()


    main()
