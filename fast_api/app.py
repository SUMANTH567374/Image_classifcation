from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO

from fast_api.model_loader import predict_image
from src.utils.logger import setup_logger

app = FastAPI()
logger = setup_logger("AppLogger", log_file="logs/app.log")

@app.get("/")
def root():
    return {"message": "Welcome to AI MedScan Inference API 🚑"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"✅ Received image: {file.filename}")
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"❌ Failed to read image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = predict_image(image)
        class_label = result["prediction"]
        confidence = float(result["confidence"])
        logger.info(f"✅ Prediction successful - Label: {class_label}, Confidence: {confidence:.4f}")
        return {"prediction": class_label, "confidence": confidence}
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Prediction failed."})
