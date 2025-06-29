import time
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.utils.logger import setup_logger
from src.utils.config_loader import load_params
from src.models.model import CNNModel

params = load_params()
paths = params["paths"]
training_params = params["training"]
label_mapping = params["labels"]

reverse_label_map = {v: k for k, v in label_mapping.items()}

logger = setup_logger("InferenceLogger", log_file=f"{paths['logs_dir']}/inference.log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load(paths["model_path"], map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((training_params["img_size"], training_params["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image):
    try:
        start_time = time.time()
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        label = reverse_label_map.get(predicted_class.item(), "Error")
        confidence_val = confidence.item()
        inference_time = time.time() - start_time

        logger.info(f"✅ Prediction: {label}, Confidence: {confidence_val:.4f}, Time: {inference_time:.2f}s")

        return {"prediction": label, "confidence": confidence_val}
    except Exception as e:
        logger.error(f"❌ Error during inference: {e}", exc_info=True)
        return {"prediction": "Error", "confidence": 0.0}
