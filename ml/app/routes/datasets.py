from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile


router = APIRouter(tags=["Datasets"])

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _dataset_preview(path: Path, limit: int = 20, sort_by: str | None = None, ascending: bool = False):
    df = pd.read_csv(path, comment="#", on_bad_lines="skip")
    original_columns = df.columns.tolist()

    if sort_by and sort_by in df.columns:
        series = pd.to_numeric(df[sort_by], errors="coerce")
        if series.notna().sum() > 0:
            df = df.assign(**{sort_by: series}).sort_values(by=sort_by, ascending=ascending, na_position="last")
        else:
            df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    preview = df.head(limit).where(pd.notna(df.head(limit)), None).to_dict(orient="records")
    return {
        "filename": path.name,
        "rows": int(len(df)),
        "columns": original_columns,
        "preview": preview,
        "sorted_by": sort_by,
        "ascending": ascending,
    }


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    destination = UPLOAD_DIR / Path(file.filename).name
    with open(destination, "wb") as handle:
        handle.write(file.file.read())

    try:
        preview = _dataset_preview(destination, limit=10)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid CSV: {exc}")

    return {
        "status": "uploaded",
        "path": str(destination),
        "dataset": preview,
    }


@router.get("/uploads")
def list_uploaded_datasets():
    files = []
    for path in sorted(UPLOAD_DIR.glob("*.csv")):
        files.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
            }
        )
    return {"count": len(files), "files": files}


@router.get("/preview")
def preview_uploaded_dataset(
    filename: str = Query(..., description="Uploaded CSV filename"),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str | None = Query(None, description="Column to sort by"),
    ascending: bool = Query(False, description="Sort ascending if true"),
):
    path = UPLOAD_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Uploaded dataset '{filename}' not found.")
    try:
        return _dataset_preview(path, limit=limit, sort_by=sort_by, ascending=ascending)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset preview: {exc}")
