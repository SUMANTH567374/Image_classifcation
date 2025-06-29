# ----------------------------------------
# File: app/schemas.py
# ----------------------------------------
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """
    Schema for model prediction response.
    """
    class_label: str = Field(..., example="Covid", description="Predicted class label")
    confidence: float = Field(..., example=0.9785, description="Prediction confidence score")
