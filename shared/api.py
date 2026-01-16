import sys

# import inference code
sys.path.append("notebook")

import os
import io
import uuid
import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from inference import Inference

# ----------------------------
# Config / Globals
# ----------------------------
out_dir = Path("../shared/out")
out_dir.mkdir(parents=True, exist_ok=True)

tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
model = Inference(config_path, compile=False)

jobs: Dict[str, str] = {}
queue: asyncio.Queue[Tuple[str, np.ndarray, np.ndarray]] = asyncio.Queue()


# ----------------------------
# Image / Mask preprocessing
# ----------------------------
async def pil_from_upload_file(u: UploadFile) -> Image.Image:
    if not (u.content_type and u.content_type.startswith("image/")):
        raise HTTPException(400, "Bad image content-type")
    data = await u.read()
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = ImageOps.exif_transpose(im)  # normalize EXIF orientation early
        return im
    except Exception:
        raise HTTPException(400, "Bad image")

def to_numpy_rgb(pil_im: Image.Image) -> np.ndarray:
    pil_im = pil_im.convert("RGB")
    return np.asarray(pil_im, dtype=np.uint8)  # (H,W,3)

def to_numpy_mask_2d(pil_mask: Image.Image, size_wh: Tuple[int, int]) -> np.ndarray:
    # Ensure mask matches image size and is 2D (H,W) boolean
    pil_mask = ImageOps.exif_transpose(pil_mask)
    pil_mask = pil_mask.convert("L")  # 1 channel
    if pil_mask.size != size_wh:
        pil_mask = pil_mask.resize(size_wh, resample=Image.NEAREST)
    mask = np.asarray(pil_mask, dtype=np.uint8)  # (H,W)
    return (mask > 0)  # bool (H,W)


# ----------------------------
# Robust "done means file complete" inference writer
# ----------------------------
def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def run_inference(job_id: str, image: np.ndarray, mask: np.ndarray) -> None:
    tmp_path = out_dir / f"{job_id}.tmp.glb"
    final_path = out_dir / f"{job_id}.glb"

    # Clean leftovers defensively
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    out = model(image, mask, seed=42)

    # Write to temp then atomically publish
    out["glb"].export(tmp_path)
    _fsync_file(tmp_path)
    os.replace(tmp_path, final_path)

    # Hard invariant
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError("GLB publish failed or produced empty file")


# ----------------------------
# Async worker
# ----------------------------
async def worker() -> None:
    while True:
        job_id, image, mask = await queue.get()
        jobs[job_id] = "running"
        try:
            await asyncio.to_thread(run_inference, job_id, image, mask)
            jobs[job_id] = "done"
        except Exception:
            jobs[job_id] = "failed"
            print(f"\n--- JOB {job_id} FAILED ---")
            print(traceback.format_exc())
            print("--- END TRACEBACK ---\n")
        finally:
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(worker())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)


# ----------------------------
# API
# ----------------------------
@app.post("/start")
async def start(
    upload_image: UploadFile = File(...),
    upload_mask: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = "queued"

    pil_img = await pil_from_upload_file(upload_image)
    pil_msk = await pil_from_upload_file(upload_mask)

    image = to_numpy_rgb(pil_img)                   # (H,W,3) uint8
    mask = to_numpy_mask_2d(pil_msk, pil_img.size)  # (H,W) bool

    await queue.put((job_id, image, mask))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    return {"status": jobs.get(job_id, "unknown")}


@app.get("/download/{job_id}")
def download(job_id: str):
    if jobs.get(job_id) != "done":
        raise HTTPException(409, "Not ready")

    path = out_dir / f"{job_id}.glb"
    if not path.exists():
        raise HTTPException(500, "Result missing")

    # Do NOT delete jobs[job_id] here; it can race with streaming.
    # Clean up with a separate reaper if you want.
    return FileResponse(
        path=path,
        media_type="model/gltf-binary",
        filename=f"{job_id}.glb",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
