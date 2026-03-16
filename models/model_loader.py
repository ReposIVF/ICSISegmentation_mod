"""
Loads and returns the YOLO detector and the TabTransformer scoring model.
"""
import torch
from torch import nn
from ultralytics import YOLO
from tab_transformer_pytorch import TabTransformer


def load_yolo(model_path: str) -> YOLO:
    """Load the YOLO segmentation/tracking model."""
    model = YOLO(model_path)
    print(f"YOLO model loaded: {model_path}")
    return model


def load_tabtransformer(model_path: str, cfg: dict, device: str) -> TabTransformer:
    """
    Instantiate and load weights for the TabTransformer blastocyst-scoring model.

    Args:
        model_path : path to the .pth weights file
        cfg        : full config dict (uses cfg['tabtransformer'] section)
        device     : 'cuda' | 'mps' | 'cpu'
    """
    arch = cfg["tabtransformer"]

    model = TabTransformer(
        categories=tuple(),
        num_continuous=arch["num_continuous"],
        dim=arch["dim"],
        dim_out=arch["dim_out"],
        depth=arch["depth"],
        heads=arch["heads"],
        attn_dropout=arch["attn_dropout"],
        ff_dropout=arch["ff_dropout"],
        mlp_hidden_mults=tuple(arch["mlp_hidden_mults"]),
        mlp_act=nn.ReLU(),
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"TabTransformer loaded: {model_path}")
    return model
