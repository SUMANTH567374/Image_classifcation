import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.model import CNNModel
from src.data.dataloader import get_dataloaders
from src.utils.logger import setup_logger
from src.utils.config_loader import load_params  # ✅ load params.yaml

# Load parameters
params = load_params()
paths = params["paths"]
training_params = params["training"]
model_params = params["model"]

# Logger setup
logger = setup_logger("TrainingLogger", log_file=os.path.join(paths["logs_dir"], "train.log"))

def train_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_loader, val_loader = get_dataloaders()
    except Exception as e:
        logger.error(f"Failed to load dataloaders: {e}")
        return

    try:
        model = CNNModel().to(device)  # ✅ No need to pass num_classes now (loaded inside model)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params["learning_rate"])

    best_val_acc = 0.0
    os.makedirs(paths["model_dir"], exist_ok=True)

    try:
        for epoch in range(training_params["num_epochs"]):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params['num_epochs']} - Training"):
                try:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                except Exception as batch_error:
                    logger.warning(f"Batch failed in epoch {epoch+1}: {batch_error}")

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), paths["model_path"])
                logger.info(f"✅ Saved best model with Val Acc = {val_acc:.4f}")

        logger.info("🏁 Training complete.")

    except Exception as train_error:
        logger.error(f"Training failed: {train_error}")


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    except Exception as eval_error:
        logger.error(f"Evaluation failed: {eval_error}")
        return 0.0, 0.0


if __name__ == "__main__":
    try:
        train_model()
    except Exception as main_error:
        logger.critical(f"Unhandled exception in training pipeline: {main_error}")
