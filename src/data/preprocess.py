import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.utils.config_loader import load_params

# Load parameters from params.yaml
params = load_params()
paths = params["paths"]
preprocessing_params = params["preprocessing"]

# Setup logger
logger = setup_logger("PreprocessingLogger", log_file=os.path.join(paths["logs_dir"], "preprocessing.log"))

def preprocess_and_save(csv_path, image_dir, output_dir, output_csv_path, img_size):
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV: {csv_path} with {len(df)} entries")
    except Exception as e:
        logger.error(f"Failed to read CSV file {csv_path}: {e}")
        return

    label_map = {label: idx for idx, label in enumerate(df['Diagnosis'].unique())}
    logger.info(f"Label mapping: {label_map}")

    processed_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path}"):
        img_name = row['image_name']
        label = label_map[row['Diagnosis']]
        img_path = os.path.join(image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            resized = transforms.Resize((img_size, img_size))(image)
            resized_path = os.path.join(output_dir, img_name)
            resized.save(resized_path)

            processed_data.append({"image_name": img_name, "label": label})
        except Exception as e:
            logger.warning(f"Could not process {img_path}: {e}")

    try:
        pd.DataFrame(processed_data).to_csv(output_csv_path, index=False)
        logger.info(f"Saved processed CSV to {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save processed CSV {output_csv_path}: {e}")

    logger.info(f"✅ Finished processing {csv_path}. Total images saved: {len(processed_data)}")


if __name__ == "__main__":
    img_size = preprocessing_params["img_size"]

    preprocess_and_save(
        csv_path=paths["train_csv_raw"],
        image_dir=paths["train_img_dir_raw"],
        output_dir=os.path.join(paths["processed_data_dir"], "train"),
        output_csv_path=paths["train_csv"],
        img_size=img_size
    )

    preprocess_and_save(
        csv_path=paths["test_csv_raw"],
        image_dir=paths["test_img_dir_raw"],
        output_dir=os.path.join(paths["processed_data_dir"], "test"),
        output_csv_path=paths["test_csv"],
        img_size=img_size
    )
