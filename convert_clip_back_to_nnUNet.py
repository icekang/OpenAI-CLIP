from CLIP import CLIPModel
import torch
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
import config as CFG
from collections import OrderedDict
from argparse import ArgumentParser


def get_nnunet_trainer():
    trainer = get_trainer_from_args(
        dataset_name_or_id=CFG.nnUNet['dataset_name_or_id'],
        configuration=CFG.nnUNet['configuration'],
        fold=CFG.nnUNet['fold'],
        trainer_name=CFG.nnUNet['trainer_name'],
        plans_identifier=CFG.nnUNet['plans_identifier'],
        device=torch.device('cpu'))
    trainer.initialize()

    return trainer


def get_clip_checkpoint():
    checkpoint = torch.load(f'logs/{CFG.experiment_name}/best.pt', map_location='cpu')
    return checkpoint


def convert_clip_to_nnunet():
    checkpoint = get_clip_checkpoint()
    encoder_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        if 'image_encoder1.' in k:
            encoder_checkpoint[k.replace('image_encoder1.model.', '')] = v

    trainer = get_nnunet_trainer()
    load_success = trainer.network.encoder.load_state_dict(encoder_checkpoint)
    print(load_success)
    trainer.save_checkpoint(f'logs/{CFG.experiment_name}/nnunet.pt')
    print("Converted CLIP model to nnU-Net model and saved to nnunet.pt")

if __name__ == "__main__":
    parser = ArgumentParser()
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
    
    convert_clip_to_nnunet()