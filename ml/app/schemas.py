from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    features: List[float]
    planet_name: Optional[str] = None

class TrainResponse(BaseModel):
    accuracy: float
    f1: float

