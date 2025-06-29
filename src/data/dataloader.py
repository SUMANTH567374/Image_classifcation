import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.logger import setup_logger
from src.utils.config_loader import load_params  # ✅ Corrected import

# Load parameters
params = load_params()
paths = params["paths"]
training_params = params["training"]

# Setup logger
logger = setup_logger("DataLoaderLogger", log_file=os.path.join(paths["logs_dir"], "data_loader.log"))

class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Validate columns
        expected_cols = {"image_name", "label"}
        if not expected_cols.issubset(set(self.data.columns)):
            raise ValueError(f"CSV must contain the columns: {expected_cols}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_name'])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        label = int(row['label'])
        return image, label


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((training_params["img_size"], training_params["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_csv = paths["train_csv"]
    test_csv = paths["test_csv"]
    train_img_dir = os.path.join(paths["processed_data_dir"], "train")
    test_img_dir = os.path.join(paths["processed_data_dir"], "test")

    train_dataset = MedicalImageDataset(train_csv, train_img_dir, transform)
    val_dataset = MedicalImageDataset(test_csv, test_img_dir, transform)

    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(val_dataset)} validation samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params["batch_size"],
        shuffle=True,
        num_workers=training_params["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params["batch_size"],
        shuffle=False,
        num_workers=training_params["num_workers"]
    )

    return train_loader, val_loader
