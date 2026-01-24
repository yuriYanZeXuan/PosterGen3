#!/usr/bin/env python3
"""
Local FastAPI server for image generation/editing.

- Generate: Z-Image (Diffusers `ZImagePipeline`) from local weights path.
- Edit: Qwen Image Edit Plus (Diffusers `QwenImageEditPlusPipeline`).

Edit results are post-processed to remove background and saved as RGBA PNG.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Hard-coded local model paths (no env vars needed).
ZIMAGE_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"
QWEN_EDIT_MODEL_ID = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_edit_2511"

# Hard-coded device placement to avoid OOM.
# Z-Image -> cuda:0, Qwen-Edit -> cuda:1 (if available).
ZIMAGE_DEVICE = "cuda:0"
QWEN_EDIT_DEVICE = "cuda:1"


app = FastAPI(title="PosterGen Local Image API", version="0.1.0")


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_out_dir(out_dir: Optional[str]) -> Path:
    d = Path(out_dir) if out_dir else (PROJECT_ROOT / "dev" / "outputs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_device(preferred: str) -> str:
    """Resolve a preferred CUDA device string, with minimal fallback."""
    if not torch.cuda.is_available():
        return "cpu"
    if preferred.startswith("cuda:"):
        try:
            idx = int(preferred.split(":")[1])
        except Exception:
            return "cuda:0"
        if torch.cuda.device_count() > idx:
            return preferred
        return "cuda:0"
    return "cuda:0"


_ZIMAGE_PIPE: Any = None
_QWEN_EDIT_PIPE: Any = None


def _load_zimage() -> Any:
    """Lazy-load Z-Image pipeline."""
    global _ZIMAGE_PIPE
    if _ZIMAGE_PIPE is not None:
        return _ZIMAGE_PIPE

    from diffusers import ZImagePipeline

    model_path = ZIMAGE_MODEL_PATH
    if not Path(model_path).exists():
        raise RuntimeError(f"Z-Image model path does not exist: {model_path}")

    device = _resolve_device(ZIMAGE_DEVICE)
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    _ZIMAGE_PIPE = pipe
    return _ZIMAGE_PIPE


def _load_qwen_edit() -> Any:
    """Lazy-load Qwen edit pipeline."""
    global _QWEN_EDIT_PIPE
    if _QWEN_EDIT_PIPE is not None:
        return _QWEN_EDIT_PIPE

    from diffusers import QwenImageEditPlusPipeline

    model_id = QWEN_EDIT_MODEL_ID
    device = _resolve_device(QWEN_EDIT_DEVICE)
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    _QWEN_EDIT_PIPE = pipe
    return _QWEN_EDIT_PIPE


def _save_image(img: Image.Image, out_dir: Path, stem: str) -> str:
    path = out_dir / f"{stem}.png"
    img.save(path)
    return str(path.resolve())


def _remove_background_rgba(img: Image.Image) -> Image.Image:
    """
    Remove background to RGBA using `rembg` only.
    """
    # If already has alpha, keep it.
    if img.mode in ("RGBA", "LA"):
        return img.convert("RGBA")

    from rembg import remove as rembg_remove  # type: ignore

    out = rembg_remove(img.convert("RGB"))
    if isinstance(out, Image.Image):
        return out.convert("RGBA")
    if isinstance(out, (bytes, bytearray)):
        return Image.open(io.BytesIO(out)).convert("RGBA")  # type: ignore[name-defined]

    raise RuntimeError(f"unexpected rembg output type: {type(out)}")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    seed: int = 42
    out_dir: Optional[str] = None
    engine: str = Field(default="zimage", description="zimage")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "zimage_device": _resolve_device(ZIMAGE_DEVICE),
        "qwen_edit_device": _resolve_device(QWEN_EDIT_DEVICE),
        "zimage_loaded": _ZIMAGE_PIPE is not None,
        "qwen_edit_loaded": _QWEN_EDIT_PIPE is not None,
    }


@app.post("/v1/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    if req.engine != "zimage":
        raise HTTPException(status_code=400, detail=f"unsupported engine for generate: {req.engine}")

    out_dir = _ensure_out_dir(req.out_dir)
    pipe = _load_zimage()

    g = torch.Generator(_resolve_device(ZIMAGE_DEVICE)).manual_seed(int(req.seed))
    with torch.inference_mode():
        out = pipe(
            prompt=req.prompt,
            height=int(req.height),
            width=int(req.width),
            num_inference_steps=int(req.num_inference_steps),
            guidance_scale=float(req.guidance_scale),
            generator=g,
        )
    img = out.images[0]

    stem = f"gen_{_now_tag()}_{uuid.uuid4().hex[:8]}"
    path = _save_image(img, out_dir, stem)
    return {"path": path, "engine": req.engine}


@app.post("/v1/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    num_inference_steps: int = Form(40),
    guidance_scale: float = Form(1.0),
    true_cfg_scale: float = Form(4.0),
    num_images_per_prompt: int = Form(1),
    seed: int = Form(0),
    out_dir: str = Form(""),
    remove_bg: bool = Form(True),
    engine: str = Form("qwen_edit"),
) -> Dict[str, Any]:
    if engine != "qwen_edit":
        raise HTTPException(status_code=400, detail=f"unsupported engine for edit: {engine}")

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    content = await image.read()
    try:
        src = Image.open(io.BytesIO(content)).convert("RGB")  # type: ignore[name-defined]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    out_dir_path = _ensure_out_dir(out_dir or None)
    pipe = _load_qwen_edit()

    with torch.inference_mode():
        out = pipe(
            image=[src],
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=torch.manual_seed(int(seed)),
            true_cfg_scale=float(true_cfg_scale),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=int(num_images_per_prompt),
        )
    edited = out.images[0]

    stem_base = f"edit_{_now_tag()}_{uuid.uuid4().hex[:8]}"
    raw_path = _save_image(edited, out_dir_path, stem_base + "_raw")

    final_img = edited
    final_path = raw_path
    if remove_bg:
        final_img = _remove_background_rgba(edited)
        final_path = _save_image(final_img, out_dir_path, stem_base + "_rgba")

    return {"path": final_path, "raw_path": raw_path, "engine": engine, "remove_bg": remove_bg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9106)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

