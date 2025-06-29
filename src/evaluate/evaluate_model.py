import os
import torch
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report

from src.models.model import CNNModel
from src.data.dataloader import get_dataloaders
from src.evaluate.metrics import evaluate_model
from src.utils.logger import setup_logger
from src.utils.config_loader import load_params

# Setup logger
logger = setup_logger("EvaluateLogger", "logs/evaluate_model.log")

def main():
    # Load params inside main to avoid global/local conflict
    params = load_params()
    paths = params["paths"]
    training = params["training"]
    model_config = params["model"]
    label_map = params["labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(
        f"Model Configuration: "
        f"use_residual={model_config.get('use_residual', True)}, "
        f"use_attention={model_config.get('use_attention', True)}, "
        f"dropout_rate={training.get('dropout_rate', 0.3)}, "
        f"num_classes={model_config.get('num_classes')}"
    )

    # Ordered class names
    class_names = [None] * len(label_map)
    for name, idx in label_map.items():
        class_names[idx] = name

    try:
        # Load validation data
        _, val_loader = get_dataloaders()

        # Load model
        model = CNNModel().to(device)
        model.load_state_dict(torch.load(paths["model_path"], map_location=device))

        # Evaluate
        acc, cm, _ = evaluate_model(model, val_loader, device, class_names)

        # Full classification report
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        # Save metrics
        os.makedirs(paths["report_dir"], exist_ok=True)
        metrics_path = os.path.join(paths["report_dir"], "metrics.json")
        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": acc,
            "confusion_matrix": np.array(cm).tolist(),
            "classification_report": report_dict
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"✅ Evaluation report saved to {metrics_path}")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
