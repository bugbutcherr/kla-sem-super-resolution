# ================================================================
# KLA HACKATHON — HIGH PERFORMANCE RTX 4060 TRAINING & SUBMISSION
# Resuming NAFNetSR from Epoch 24 to 300+
# ================================================================
import os
import sys
import glob
import copy
import time
import zipfile
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Enable Tensor Cores on Ada Lovelace RTX 4060 ────────────────
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ── Model Architecture (width=48, matches checkpoint) ───────────
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
        m = self.middle(e4)
        x = self.dec_level3(self.up3(m) + e3)
        x = self.dec_level2(self.up2(x) + e2)
        x = self.dec_level1(self.up1(x) + e1)
        x = self.dec_level0(x)
        return self.tail(x)

# ── High-Performance In-Place GPU EMA ───────────────────────────
class GPU_EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, target_model):
        target_model.load_state_dict(self.shadow)

# ── GPU Batch Augmentation ──────────────────────────────────────
def augment_batch(lr_batch, hr_batch):
    if torch.rand(1).item() > 0.5:
        lr_batch = torch.flip(lr_batch, dims=[3])
        hr_batch = torch.flip(hr_batch, dims=[3])
    if torch.rand(1).item() > 0.5:
        lr_batch = torch.flip(lr_batch, dims=[2])
        hr_batch = torch.flip(hr_batch, dims=[2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        lr_batch = torch.rot90(lr_batch, k, dims=[2, 3])
        hr_batch = torch.rot90(hr_batch, k, dims=[2, 3])
    return lr_batch.contiguous(), hr_batch.contiguous()

# ── 8-Fold Test-Time Augmentation (TTA) ─────────────────────────
def predict_8fold_tta(model, x):
    preds = []
    for k in range(4):
        r = torch.rot90(x, k, [2, 3])
        o = model(r)
        preds.append(torch.rot90(o, -k, [2, 3]))
        f = torch.flip(r, [3])
        of = model(f)
        preds.append(torch.rot90(torch.flip(of, [3]), -k, [2, 3]))
    return torch.clamp(torch.stack(preds).mean(0), 0.0, 1.0)

def log_msg(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ── Main Pipeline ───────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = Path(__file__).resolve().parent
    log_file = base_dir / "training_log.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Training Log ===\n")

    log_msg("================================================================", log_file)
    log_msg("🚀 KLA HACKATHON: High-Performance GPU Training & Submission", log_file)
    log_msg(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", log_file)
    log_msg(f"⚡ TF32 Tensor Cores: Enabled | PyTorch: {torch.__version__}", log_file)
    log_msg("================================================================", log_file)

    data_dir = base_dir / "data" / "KLA Hackathon"
    if not data_dir.exists():
        data_dir = base_dir / "data"

    train_noisy_dir = data_dir / "train" / "NoisyLR"
    train_gt_dir = data_dir / "train" / "GT"
    test_noisy_dir = data_dir / "NoisyLR"

    all_noisy = sorted(list(train_noisy_dir.glob("*.npy")))
    all_gt = sorted(list(train_gt_dir.glob("*.npy")))
    test_files = sorted(list(test_noisy_dir.glob("*.npy")))

    log_msg(f"Found {len(all_noisy)} training pairs, {len(test_files)} test images.", log_file)

    # 90% train / 10% validation split (2880 / 320)
    split = int(len(all_noisy) * 0.9)
    log_msg("Preloading contiguous tensor arrays into RAM...", log_file)
    
    train_lr_list = [np.load(f).astype(np.float32) for f in all_noisy[:split]]
    train_hr_list = [np.load(f).astype(np.float32) for f in all_gt[:split]]
    val_lr_list   = [np.load(f).astype(np.float32) for f in all_noisy[split:]]
    val_hr_list   = [np.load(f).astype(np.float32) for f in all_gt[split:]]

    train_lr_tensor = torch.from_numpy(np.stack(train_lr_list, axis=0)).unsqueeze(1) # [2880, 1, 128, 128]
    train_hr_tensor = torch.from_numpy(np.stack(train_hr_list, axis=0)).unsqueeze(1) # [2880, 1, 256, 256]
    val_lr_tensor   = torch.from_numpy(np.stack(val_lr_list, axis=0)).unsqueeze(1)   # [320, 1, 128, 128]
    val_hr_tensor   = torch.from_numpy(np.stack(val_hr_list, axis=0)).unsqueeze(1)   # [320, 1, 256, 256]
    
    log_msg(f"✅ Tensors loaded: Train LR {train_lr_tensor.shape}, HR {train_hr_tensor.shape}", log_file)
    log_msg(f"✅ Validation: {len(val_lr_list)} pairs", log_file)

    model = NAFNetSR(width=48).to(device)
    val_model = NAFNetSR(width=48).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log_msg(f"✅ Model: NAFNetSR ({param_count/1e6:.2f}M parameters)", log_file)

    # Load transferred checkpoint
    ckpt_path = base_dir / "kla_best_model.pth"
    save_path = base_dir / "best_model_rtx.pth"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_save = models_dir / "best_model.pth"

    start_ep = 0
    best_psnr = 0.0
    TOTAL = 300

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4, fused=True)

    if ckpt_path.exists():
        log_msg(f"Loading transferred checkpoint: {ckpt_path}...", log_file)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        start_ep = ck.get('epoch', 24) + 1
        best_psnr = ck.get('best_psnr', 27.82)
        if 'optimizer_state_dict' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer_state_dict'])
            except Exception as e:
                log_msg(f"Note on optimizer load: {e}", log_file)
        for pg in optimizer.param_groups:
            pg['lr'] = 1e-4
        log_msg(f"✅ Successfully resumed from Epoch {start_ep} | Baseline Best PSNR: {best_psnr:.2f} dB", log_file)

    ema = GPU_EMA(model, decay=0.999)

    def get_lr(ep):
        warmup = 10
        rel = ep - start_ep
        if rel < warmup:
            return 1e-4 + (1e-3 - 1e-4) * (rel / max(warmup, 1))
        progress = (rel - warmup) / max(TOTAL - start_ep - warmup, 1)
        return 1e-6 + 0.5 * (1e-3 - 1e-6) * (1.0 + np.cos(np.pi * progress))

    # Fast micro-batch size with gradient accumulation
    micro_batch = 8
    accum_steps = 2
    n_train = len(train_lr_tensor)
    n_micro_batches = n_train // micro_batch
    n_val = len(val_lr_tensor)
    val_batch_size = 16

    log_msg(f"\n🚀 Training on RTX 4060: Epochs {start_ep} -> {TOTAL} | Effective Batch: {micro_batch * accum_steps} | Charbonnier Loss\n", log_file)

    for epoch in range(start_ep, TOTAL):
        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        tloss = 0.0
        t0 = time.time()

        perm = torch.randperm(n_train)
        optimizer.zero_grad(set_to_none=True)

        for m_idx in range(n_micro_batches):
            idx = perm[m_idx * micro_batch : (m_idx + 1) * micro_batch]
            lr_b = train_lr_tensor[idx].to(device, non_blocking=True)
            hr_b = train_hr_tensor[idx].to(device, non_blocking=True)
            
            lr_b, hr_b = augment_batch(lr_b, hr_b)
            
            sr = model(lr_b)
            loss = torch.mean(torch.sqrt((sr - hr_b) ** 2 + 1e-6))
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss_scaled = loss / accum_steps
                loss_scaled.backward()
                tloss += loss.item()
                
            if (m_idx + 1) % accum_steps == 0 or (m_idx + 1) == n_micro_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

        epoch_time = time.time() - t0
        avg_loss = tloss / n_micro_batches

        # Validation on EMA model
        ema.apply(val_model)
        val_model.eval()
        
        psnrs = []
        with torch.no_grad():
            for v_idx in range(0, n_val, val_batch_size):
                v_lr = val_lr_tensor[v_idx : v_idx + val_batch_size].to(device, non_blocking=True)
                v_hr = val_hr_tensor[v_idx : v_idx + val_batch_size].to(device, non_blocking=True)
                pred = torch.clamp(val_model(v_lr), 0.0, 1.0)
                
                mse = torch.mean((pred - v_hr) ** 2, dim=[1, 2, 3])
                batch_psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
                psnrs.extend(batch_psnr.cpu().tolist())

        vp = float(np.mean(psnrs)) if psnrs else 0.0
        flag = ""
        if vp > best_psnr and vp < 60:
            best_psnr = vp
            flag = "🔥 [NEW BEST!]"
            save_payload = {
                'epoch': epoch,
                'model_state_dict': val_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(save_payload, save_path)
            torch.save(save_payload, best_model_save)

        msg = f"Epoch [{epoch + 1:3d}/{TOTAL:3d}] ({epoch_time:.1f}s) | Loss: {avg_loss:.5f} | Val PSNR: {vp:.2f} dB | LR: {lr:.2e} {flag}"
        log_msg(msg, log_file)

    log_msg(f"\n================================================================", log_file)
    log_msg(f"✅ Training Finished! Top Val PSNR: {best_psnr:.2f} dB", log_file)
    log_msg(f"Checkpoint saved to: {save_path} and {best_model_save}", log_file)
    log_msg(f"================================================================", log_file)

    # ── 8-Fold TTA Test Inference ─────────────────────────────────
    log_msg(f"\n🔮 Generating Final Predictions with 8-Fold TTA on {len(test_files)} test images...", log_file)
    preds_dir = base_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = torch.load(best_model_save if best_model_save.exists() else ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        for fp in tqdm(test_files, desc="TTA Inference"):
            arr = np.load(fp).astype(np.float32)
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            out_tensor = predict_8fold_tta(model, t).squeeze().cpu().numpy()
            out_sanitized = np.nan_to_num(out_tensor, nan=0.0, posinf=1.0, neginf=0.0)
            out_sanitized = np.clip(out_sanitized, 0.0, 1.0).astype(np.float32)
            
            assert out_sanitized.shape == (256, 256), f"Wrong shape: {out_sanitized.shape}"
            assert not np.isnan(out_sanitized).any(), "NaN found!"
            assert not np.isinf(out_sanitized).any(), "Inf found!"
            
            out_fp = preds_dir / fp.name
            np.save(out_fp, out_sanitized)

    # Create submission zip
    zip_path = base_dir / "predictions_submission.zip"
    log_msg(f"\n📦 Packaging {len(test_files)} predictions into {zip_path}...", log_file)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(list(preds_dir.glob("*.npy"))):
            zf.write(f, arcname=f.name)
    log_msg(f"✅ Submission ZIP created: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)", log_file)

    # ── Package Standalone team_name Folder ───────────────────────
    log_msg(f"\n📁 Creating standalone submission folder 'team_name/'...", log_file)
    team_dir = base_dir / "team_name"
    team_models_dir = team_dir / "models"
    team_models_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_model_save if best_model_save.exists() else ckpt_path, team_models_dir / "best_model.pth")
    log_msg("✅ Standalone 'team_name/' package updated successfully!", log_file)

if __name__ == "__main__":
    main()
