"""Default run configuration and ImageNet normalisation constants."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Mutable defaults for training / data (copy if you need to change per run).
DEFAULT_CFG: dict = {
    "data_dir": "data/montgomery/MontgomerySet",
    "img_size": 256,
    "freeze_epochs": 5,
    "num_epochs": 30,
    "encoder_lr": 1e-4,
    "decoder_lr": 1e-3,
    "batch_size": 8,
    "seed": 42,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "device": "cuda",  # overwritten at runtime if no GPU
    "ckpt_path": "resnet18_unet_best.pth",
}
