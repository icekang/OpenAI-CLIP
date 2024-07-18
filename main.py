import os
import gc
import numpy as np
import pandas as pd
import torch.amp
from tqdm import tqdm

import torch
from torch import nn

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr
from torch.utils.tensorboard import SummaryWriter

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
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image1"].values,
        dataframe["image2"].values,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, scaler):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k not in ["caption", "image_path1", "image_path2", "image_z_index"]}
            loss = model(batch)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == "batch":
            lr_scheduler.step()
        optimizer.zero_grad()

        count = batch["image1"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k not in ["caption", "image_path1", "image_path2", "image_z_index"]}
            loss = model(batch, mode="valid")

        count = batch["image1"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    writer = SummaryWriter()
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
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, scaler)
        writer.add_scalar("Loss/train", train_loss.avg, epoch)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            writer.add_scalar("Loss/valid", valid_loss.avg, epoch)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
