from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter()

AWARENESS_FILE = Path("app/system/awareness_state.json")

@router.get("/awareness")
async def get_awareness():
    """
    Return the current system awareness state (last dataset, model hash, etc.)
    """
    if not AWARENESS_FILE.exists():
        raise HTTPException(status_code=404, detail="Awareness state not found")
    try:
        with open(AWARENESS_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

