from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    features: List[float]
    planet_name: Optional[str] = None

class TrainResponse(BaseModel):
    accuracy: float
    f1: float


class ChatRequest(BaseModel):
    message: str
    limit: int = 5
    session_id: Optional[str] = None
    page: Optional[int] = None
    use_openai: bool = False
    active_planet: Optional[str] = None
