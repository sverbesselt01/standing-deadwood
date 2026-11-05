import glob
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from safetensors.torch import load_file, save_file

# --- CONFIGURATION ---
DATA_DIR = "C:/Users/sebastiaan_verbessel/Documents/test_senseforest/my_dataset/tiles_1024/patches"  # Directory containing .png and _mask.png files
INITIAL_WEIGHTS_PATH = "model/segformer_b5_full_epoch_100.safetensors"
OUTPUT_WEIGHTS_PATH = "model/segformer_b5_finetuned.safetensors"

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (1024, 1024)


# --- DATASET ---
class PngPatchDataset(Dataset):
    def __init__(self, data_dir: str, target_size=(1024, 1024)):
        self.target_size = target_size
        self.pairs = []

        # Find all .png files that do NOT end with _mask.png
        all_pngs = glob.glob(os.path.join(data_dir, "*.png"))
        image_candidates = [f for f in all_pngs if not f.endswith("_mask.png")]

        for img_path in sorted(image_candidates):
            # Replace trailing .png with _mask.png
            mask_path = img_path[:-4] + "_mask.png"
            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Missing mask for {img_path}, skipping.")

        assert len(self.pairs) > 0, f"No valid image/mask pairs found in {data_dir}"

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL uint8 [0, 255] -> float32 [0.0, 1.0]
            transforms.Resize(self.target_size, antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize(
                self.target_size, 
                interpolation=transforms.InterpolationMode.NEAREST
            ),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # 1. Load RGB Image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_tensor = self.img_transforms(img)

        # 2. Load Mask (single-channel)
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask = self.mask_transforms(mask)
            mask_np = np.array(mask, dtype=np.float32)

        # Binarize mask to {0.0, 1.0} and set shape to [1, H, W]
        mask_np = (mask_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return img_tensor, mask_tensor


# --- MODEL INITIALIZATION ---
def get_model(weights_path: str, device: torch.device) -> nn.Module:
    model = smp.Unet(
        encoder_name="mit_b5",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(memory_format=torch.channels_last)

    if os.path.exists(weights_path):
        state_dict = load_file(weights_path, device=str(device))
        cleaned_state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned_state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights file not found at {weights_path}, starting from default encoder.")

    return model.to(device=device, memory_format=torch.channels_last)


# --- TRAINING ---
def main():
    dataset = PngPatchDataset(DATA_DIR, target_size=TARGET_SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )

    model = get_model(INITIAL_WEIGHTS_PATH, DEVICE)
    model.train()

    # Focal Tversky Loss (alpha=0.1, beta=0.9, gamma=2.0)
    loss_fn = smp.losses.TverskyLoss(
        mode="binary",
        alpha=0.1,
        beta=0.9,
        gamma=2.0,
        from_logits=True,
        smooth=1.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    print(f"Training on {len(dataset)} paired PNG patches...")

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0

        for images, masks in loader:
            images = images.to(device=DEVICE, memory_format=torch.channels_last, non_blocking=True)
            masks = masks.to(device=DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Focal Tversky Loss: {epoch_loss:.4f}")

    # Save fine-tuned checkpoint
    os.makedirs(os.path.dirname(OUTPUT_WEIGHTS_PATH) or ".", exist_ok=True)
    save_file(model.state_dict(), OUTPUT_WEIGHTS_PATH)
    print(f"Finetuned model saved to {OUTPUT_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()