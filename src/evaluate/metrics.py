# src/evaluate/metrics.py

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger("EvaluateLogger", "logs/metrics.log")

def evaluate_model(model, dataloader, device, class_names=None):
    """
    Evaluates a model and returns accuracy, confusion matrix, and classification report.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Validation/test DataLoader.
        device (torch.device): 'cuda' or 'cpu'.
        class_names (list): Class label names.

    Returns:
        (float, np.ndarray, str): accuracy, confusion matrix, classification report
    """
    model.eval()
    y_true, y_pred = [], []

    try:
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                except Exception as batch_error:
                    logger.warning(f"⚠️ Skipped batch due to error: {batch_error}")

        if not y_true or not y_pred:
            logger.error("❌ No predictions made. Evaluation failed.")
            return 0.0, np.array([]), ""

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

        logger.info(f"✅ Accuracy: {acc:.4f}")
        logger.info(f"📊 Confusion Matrix:\n{cm}")
        logger.info(f"🧾 Classification Report:\n{report}")

        return acc, cm, report

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return 0.0, np.array([]), ""
