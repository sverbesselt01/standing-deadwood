import os
import torch
import numpy as np

# Import the trainer directly from your package/module structure
from model.trainer import DeadwoodTrainer

# Automatically detect available compute device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    # Pin to the single available GPU (GPU 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEFAULT_CONFIG = {
    "experiment_name": "unet_test_accelerate", # aan te passen
    "device": DEVICE,
    "use_wandb": False,
    "save_checkpoint": True,
    "epochs": 10,
    "val_every": 1,
    "no_folds": 3,
    "run_fold": -1,
    "batch_size": 8,
    "epoch_train_samples": 128,
    "epoch_val_samples": 128,
    "test_size": 0, # 0.2 (can be 20%)
    "balancing_factor": 1,
    "pos_weight": 12,
    "bce_weight": 0.9,
    "bins": np.arange(1, 28, 1), # the resolution bins of my gsd (going from 1cm up to 28 cm, bin size = 1 cm)
    "amp": torch.cuda.is_available(),  # Enable AMP only when GPU is present
    "learning_rate": 1e-5,
    "weight_decay": 1e-8,
    "momentum": 0.999,
    "lr_patience": 5,
    "num_workers": 0,  # Reduced to 0 since we are working on a windows computer
    "gradient_clipping": 1.0,
    "experiments_dir": "./experiments",
    "images_dir": "C:/Users/sebastiaan_verbessel/Documents/test_senseforest/my_dataset/tiles_1024",
    "register_file": "C:/Users/sebastiaan_verbessel/Documents/test_senseforest/my_dataset/tiles_1024/dataset_metadata_1024.csv",
    "random_seed": 10,
    "loss": "bce",
    "gradient_accumulation": 1,
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "warmup_epochs": 2,
    "warmup_start_lr": 1e-9
}

# DEFAULT_CONFIG = { 
#     "experiment_name": "segformer_b5_test",
#     "use_wandb": "true",
#     "save_checkpoint": "true",
#     "epochs": 200,
#     "val_every": 15,
#     "no_folds": 1, # no cross-validation
#     "batch_size": 4,
#     "gradient_accumulation":1,
#     "epoch_train_samples": 128,
#     "epoch_val_samples": 128, 
#     "test_size": 0.2, # set 20% of your data as test set
#     "balancing_factor": 1,
#     "num_workers": 0,  # Reduced to 0 since we are working on a windows computer
#     "learning_rate": 1e-5,
#     "weight_decay": 1e-4,
#     "experiments_dir": "./experiments",
#     "images_dir": "./data/tiles",
#     "register_file": "C:/Users/sebastiaan_verbessel/Documents/test_senseforest/my_dataset/dataset_metadata.csv",
#     "random_seed": 11,
#     "encoder_name": "mit_b5",
#     "encoder_weights": "imagenet",
#     "loss": "tverskyfocal",
#     "alpha": 0.1,
#     "beta": 0.9,
#     "gamma": 2,
#     "warmup_epochs": 15,
#     "warmup_start_lr": 1e-9
# }

def main() -> None:
    print(f"Using device: {DEVICE}")

    config = dict(DEFAULT_CONFIG)
    trainer = DeadwoodTrainer(run_name=config["experiment_name"], config=config)
    trainer.run()


if __name__ == "__main__":
    main()