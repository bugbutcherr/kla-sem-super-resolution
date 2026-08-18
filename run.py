# ================================================================
# KLA Hackathon - SEM Super-Resolution Inference CLI
# Multi-Checkpoint 16-Pass Ensemble (2 Models x 8-Fold TTA)
# Usage: python run.py <input-directory> <output-directory>
# ================================================================
import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw = c * 2
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg1   = SimpleGate()
        self.sca   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.norm1 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(c, dw, 1)
        self.sg2   = SimpleGate()
        self.conv5 = nn.Conv2d(dw // 2, c, 1)
        self.norm2 = nn.GroupNorm(1, c)

    def forward(self, x):
        r = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = r + y

        r = x
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = r + y
        return x

class NAFNetSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=48,
                 enc_blocks=[2, 2, 4, 8], dec_blocks=[2, 2, 2, 2], middle_blocks=12, scale=2):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.enc_level1 = nn.Sequential(*[NAFBlock(width) for _ in range(enc_blocks[0])])
        self.down1 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.enc_level2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(enc_blocks[1])])
        self.down2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.enc_level3 = nn.Sequential(*[NAFBlock(width * 4) for _ in range(enc_blocks[2])])
        self.down3 = nn.Conv2d(width * 4, width * 8, 2, stride=2)
        self.enc_level4 = nn.Sequential(*[NAFBlock(width * 8) for _ in range(enc_blocks[3])])

        self.middle = nn.Sequential(*[NAFBlock(width * 8) for _ in range(middle_blocks)])

        self.up3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
        self.dec_level3 = nn.Sequential(*[NAFBlock(width * 4) for _ in range(dec_blocks[0])])
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec_level2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(dec_blocks[1])])
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec_level1 = nn.Sequential(*[NAFBlock(width) for _ in range(dec_blocks[2])])
        self.dec_level0 = nn.Sequential(*[NAFBlock(width) for _ in range(dec_blocks[3])])

        self.tail = nn.Sequential(
            nn.Conv2d(width, width * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, out_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.intro(x)
        e1 = self.enc_level1(x); x = self.down1(e1)
        e2 = self.enc_level2(x); x = self.down2(e2)
        e3 = self.enc_level3(x); x = self.down3(e3)
        e4 = self.enc_level4(x)
        m  = self.middle(e4)
        x = self.dec_level3(self.up3(m)  + e3)
        x = self.dec_level2(self.up2(x)  + e2)
        x = self.dec_level1(self.up1(x)  + e1)
        x = self.dec_level0(x)
        return self.tail(x)

def predict_8fold_tta(model, tensor_in):
    """8-Fold Test-Time Augmentation (D4 dihedral group)."""
    preds = []
    for k in range(4):
        r = torch.rot90(tensor_in, k, [2, 3])
        o = model(r)
        preds.append(torch.rot90(o, -k, [2, 3]))

        f = torch.flip(r, [3])
        of = model(f)
        preds.append(torch.rot90(torch.flip(of, [3]), -k, [2, 3]))
    return torch.clamp(torch.stack(preds).mean(0), 0.0, 1.0)

def load_model(ckpt_path, device):
    """Load a NAFNetSR model from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    
    if 'intro.weight' in state_dict:
        width = state_dict['intro.weight'].shape[0]
    else:
        width = 48
        
    model = NAFNetSR(width=width).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="KLA Hackathon SEM Super-Resolution Inference CLI")
    parser.add_argument("input_dir", type=str, help="Path to input directory containing .npy files")
    parser.add_argument("output_dir", type=str, help="Path to output directory to save restored .npy files")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Running inference on device: {device}")

    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"

    checkpoint_names = ["checkpoint_mid.pth", "best_model.pth"]
    models = []
    for ckpt_name in checkpoint_names:
        ckpt_path = models_dir / ckpt_name
        if ckpt_path.exists():
            print(f"Loading {ckpt_name}...")
            models.append(load_model(ckpt_path, device))
        else:
            print(f"[INFO] {ckpt_name} not found, skipping.")

    if not models:
        raise FileNotFoundError(f"No model weights found in {models_dir}")

    total_passes = len(models) * 8
    print(f"Ensemble: {len(models)} checkpoint(s) x 8-Fold TTA = {total_passes} predictions averaged per image")

    input_files = sorted(list(input_dir.glob("*.npy")))
    print(f"Found {len(input_files)} .npy files in {input_dir}")

    with torch.no_grad():
        for fp in tqdm(input_files, desc="Restoring SEM Images"):
            arr = np.load(fp).astype(np.float32)
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            all_preds = [predict_8fold_tta(m, t) for m in models]
            final = torch.stack(all_preds).mean(0)

            out = np.nan_to_num(final.squeeze().cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)

            np.save(output_dir / fp.name, out)

    print(f"Successfully wrote {len(input_files)} predictions to {output_dir}")

if __name__ == "__main__":
    main()
