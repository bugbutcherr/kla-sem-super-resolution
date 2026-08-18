"""
================================================================
KLA HACKATHON — MULTI-CHECKPOINT ENSEMBLE INFERENCE
Combines mid-training + final checkpoint with 8-Fold TTA
Usage: python ensemble_inference.py
================================================================
"""
import sys
import zipfile
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Model Architecture (must match training) ─────────────────────
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1); return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw = c * 2
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg1   = SimpleGate()
        self.sca   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw//2, dw//2, 1))
        self.conv3 = nn.Conv2d(dw//2, c, 1)
        self.norm1 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(c, dw, 1)
        self.sg2   = SimpleGate()
        self.conv5 = nn.Conv2d(dw//2, c, 1)
        self.norm2 = nn.GroupNorm(1, c)

    def forward(self, x):
        r = x
        y = self.norm1(x)
        y = self.conv1(y); y = self.conv2(y); y = self.sg1(y)
        y = y * self.sca(y); y = self.conv3(y)
        x = r + y
        r = x
        y = self.norm2(x); y = self.conv4(y); y = self.sg2(y); y = self.conv5(y)
        return r + y

class NAFNetSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=48,
                 enc_blocks=[2,2,4,8], dec_blocks=[2,2,2,2], middle_blocks=12, scale=2):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.enc_level1 = nn.Sequential(*[NAFBlock(width) for _ in range(enc_blocks[0])])
        self.down1 = nn.Conv2d(width, width*2, 2, stride=2)
        self.enc_level2 = nn.Sequential(*[NAFBlock(width*2) for _ in range(enc_blocks[1])])
        self.down2 = nn.Conv2d(width*2, width*4, 2, stride=2)
        self.enc_level3 = nn.Sequential(*[NAFBlock(width*4) for _ in range(enc_blocks[2])])
        self.down3 = nn.Conv2d(width*4, width*8, 2, stride=2)
        self.enc_level4 = nn.Sequential(*[NAFBlock(width*8) for _ in range(enc_blocks[3])])
        self.middle = nn.Sequential(*[NAFBlock(width*8) for _ in range(middle_blocks)])
        self.up3 = nn.ConvTranspose2d(width*8, width*4, 2, stride=2)
        self.dec_level3 = nn.Sequential(*[NAFBlock(width*4) for _ in range(dec_blocks[0])])
        self.up2 = nn.ConvTranspose2d(width*4, width*2, 2, stride=2)
        self.dec_level2 = nn.Sequential(*[NAFBlock(width*2) for _ in range(dec_blocks[1])])
        self.up1 = nn.ConvTranspose2d(width*2, width, 2, stride=2)
        self.dec_level1 = nn.Sequential(*[NAFBlock(width) for _ in range(dec_blocks[2])])
        self.dec_level0 = nn.Sequential(*[NAFBlock(width) for _ in range(dec_blocks[3])])
        self.tail = nn.Sequential(
            nn.Conv2d(width, width*(scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, out_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.intro(x)
        e1 = self.enc_level1(x); x = self.down1(e1)
        e2 = self.enc_level2(x); x = self.down2(e2)
        e3 = self.enc_level3(x); x = self.down3(e3)
        e4 = self.enc_level4(x)
        m = self.middle(e4)
        x = self.dec_level3(self.up3(m) + e3)
        x = self.dec_level2(self.up2(x) + e2)
        x = self.dec_level1(self.up1(x) + e1)
        x = self.dec_level0(x)
        return self.tail(x)

# ── 8-Fold TTA ───────────────────────────────────────────────────
def predict_8fold_tta(model, x):
    preds = []
    for k in range(4):
        r = torch.rot90(x, k, [2, 3])
        preds.append(torch.rot90(model(r), -k, [2, 3]))
        f = torch.flip(r, [3])
        preds.append(torch.rot90(torch.flip(model(f), [3]), -k, [2, 3]))
    return torch.clamp(torch.stack(preds).mean(0), 0.0, 1.0)

def load_model(ckpt_path, device):
    model = NAFNetSR(width=48).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    psnr = ckpt.get('best_psnr', 0.0)
    epoch = ckpt.get('epoch', 0)
    print(f"  Loaded: {Path(ckpt_path).name} | Epoch {epoch} | PSNR {psnr:.4f} dB")
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"

    print("================================================================")
    print("KLA HACKATHON - MULTI-CHECKPOINT ENSEMBLE INFERENCE")
    print(f"Device: {device}")
    print("================================================================")

    # ── Load all available checkpoints ─────────────────────────
    checkpoints = []
    for ckpt_name in ["checkpoint_mid.pth", "best_model.pth"]:
        ckpt_path = models_dir / ckpt_name
        if ckpt_path.exists():
            print(f"\nLoading {ckpt_name}...")
            m = load_model(ckpt_path, device)
            checkpoints.append(m)
        else:
            print(f"  [SKIP] {ckpt_name} not found")

    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return

    print(f"\nEnsemble: {len(checkpoints)} checkpoint(s) × 8-Fold TTA = {len(checkpoints)*8} total predictions averaged")

    # ── Load test images ─────────────────────────────────────────
    data_dir = base_dir / "data" / "KLA Hackathon"
    test_noisy_dir = data_dir / "NoisyLR"
    test_files = sorted(list(test_noisy_dir.glob("*.npy")))
    print(f"\nTest images found: {len(test_files)}")

    preds_dir = base_dir / "predictions_ensemble"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # ── Run ensemble inference ────────────────────────────────────
    print(f"\nRunning Ensemble Inference on {len(test_files)} test images...")
    with torch.no_grad():
        for i, fp in enumerate(test_files):
            arr = np.load(fp).astype(np.float32)
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            # Average TTA predictions from all checkpoints
            all_preds = []
            for model in checkpoints:
                tta_pred = predict_8fold_tta(model, t)  # [1, 1, 256, 256]
                all_preds.append(tta_pred)

            # Final ensemble average
            final = torch.stack(all_preds).mean(0)
            out = torch.clamp(final, 0.0, 1.0).squeeze().cpu().numpy()

            # Sanitize
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)

            # Validate
            assert out.shape == (256, 256), f"Wrong shape: {out.shape}"
            assert not np.isnan(out).any(), "NaN found!"
            assert not np.isinf(out).any(), "Inf found!"

            np.save(preds_dir / fp.name, out)

            if (i+1) % 50 == 0 or i == 0:
                print(f"  [{i+1:3d}/{len(test_files)}] Processed: {fp.name}")

    print(f"\nAll {len(test_files)} images processed successfully!")

    # ── Compute validation PSNR (optional sanity check) ──────────
    val_dir = base_dir / "data" / "KLA Hackathon" / "train"
    val_lr_files = sorted(list((val_dir / "NoisyLR").glob("*.npy")))[2880:]
    val_gt_files = sorted(list((val_dir / "GT").glob("*.npy")))[2880:]

    if val_lr_files:
        print(f"\nComputing ensemble validation PSNR on {len(val_lr_files)} val images...")
        psnrs = []
        with torch.no_grad():
            for n_p, g_p in zip(val_lr_files[:50], val_gt_files[:50]):  # Quick 50-sample check
                n_arr = np.load(n_p).astype(np.float32)
                g_arr = np.load(g_p).astype(np.float32)
                t = torch.from_numpy(n_arr).unsqueeze(0).unsqueeze(0).to(device)

                all_preds = []
                for model in checkpoints:
                    all_preds.append(predict_8fold_tta(model, t))
                final = torch.stack(all_preds).mean(0)
                out = torch.clamp(final, 0.0, 1.0).squeeze().cpu().numpy()

                mse = np.mean((out - g_arr)**2)
                psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
                psnrs.append(psnr)

        print(f"  Ensemble Val PSNR (50-sample): {np.mean(psnrs):.4f} dB")

    # ── Package into submission ZIP ──────────────────────────────
    zip_path = base_dir / "predictions_ensemble_submission.zip"
    print(f"\nPackaging {len(test_files)} predictions into {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(preds_dir.glob("*.npy")):
            zf.write(fp, fp.name)
    print(f"ZIP created: {zip_path.name}")

    # ── Also update team_name/models with best checkpoint ────────
    team_model = base_dir / "team_name" / "models" / "best_model.pth"
    shutil.copy2(models_dir / "best_model.pth", team_model)
    print(f"Updated team_name/models/best_model.pth")

    print("\n================================================================")
    print(f"ENSEMBLE INFERENCE COMPLETE!")
    print(f"  Checkpoints Used: {len(checkpoints)}")
    print(f"  TTA Folds Per Model: 8")
    print(f"  Total Predictions Averaged Per Image: {len(checkpoints)*8}")
    print(f"  Submission ZIP: {zip_path}")
    print("================================================================")

if __name__ == "__main__":
    main()
