#!/usr/bin/env python3
"""
Local FastAPI server for image generation/editing (PosterGen3).

Endpoints:
- GET  /health
- POST /v1/generate   (JSON)  -> {"path": "..."}
- POST /v1/edit       (form+file) -> {"path": "...", "raw_path": "..."}

Notes:
- Z-Image runs on cuda:0
- Qwen Edit runs on cuda:1 (if available)
- Background removal supports:
  - OpenCV flood-fill (connected region)
  - RMBG-2.0 (briaai/RMBG-2.0) segmentation model
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

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Hard-coded local model paths (no env vars needed).
ZIMAGE_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"
QWEN_EDIT_MODEL_ID = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_edit_2511"

# Hard-coded device placement (avoid OOM).
ZIMAGE_DEVICE = "cuda:0"
QWEN_EDIT_DEVICE = "cuda:1"

# Qwen edit defaults
QWEN_EDIT_NEGATIVE_PROMPT = (
    "text, watermark, logo, signature, subtitle, caption, "
    "lowres, blurry, out of focus, jpeg artifacts, noise, low quality, "
    "deformed, disfigured, bad anatomy, bad proportions, "
    "extra limbs, extra fingers, missing fingers, mutated hands"
)

# RMBG-2.0 background removal (HuggingFace). Change to local path if needed.
RMBG_MODEL_ID = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/rmbg2"
# Prefer running RMBG on cuda:0 to avoid contending with Qwen Edit on cuda:1.
RMBG_DEVICE = "cuda:0"


app = FastAPI(title="PosterGen Local Image API", version="0.2.0")


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_out_dir(out_dir: Optional[str]) -> Path:
    d = Path(out_dir) if out_dir else (PROJECT_ROOT / "dev" / "outputs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_device(preferred: str) -> str:
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
_RMBG_MODEL: Any = None
_RMBG_TRANSFORM: Any = None


def _load_zimage() -> Any:
    global _ZIMAGE_PIPE
    if _ZIMAGE_PIPE is not None:
        return _ZIMAGE_PIPE

    from diffusers import ZImagePipeline

    if not Path(ZIMAGE_MODEL_PATH).exists():
        raise RuntimeError(f"Z-Image model path does not exist: {ZIMAGE_MODEL_PATH}")

    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(_resolve_device(ZIMAGE_DEVICE))
    _ZIMAGE_PIPE = pipe
    return _ZIMAGE_PIPE


def _load_qwen_edit() -> Any:
    global _QWEN_EDIT_PIPE
    if _QWEN_EDIT_PIPE is not None:
        return _QWEN_EDIT_PIPE

    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(QWEN_EDIT_MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(_resolve_device(QWEN_EDIT_DEVICE))
    pipe.set_progress_bar_config(disable=None)
    _QWEN_EDIT_PIPE = pipe
    return _QWEN_EDIT_PIPE


def _load_rmbg() -> tuple[Any, Any]:
    """Lazy-load RMBG-2.0 segmentation model and its transform."""
    global _RMBG_MODEL, _RMBG_TRANSFORM
    if _RMBG_MODEL is not None and _RMBG_TRANSFORM is not None:
        return _RMBG_MODEL, _RMBG_TRANSFORM

    device = _resolve_device(RMBG_DEVICE)
    model = AutoModelForImageSegmentation.from_pretrained(RMBG_MODEL_ID, trust_remote_code=True)
    torch.set_float32_matmul_precision("high")
    model.to(device)
    model.eval()

    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    _RMBG_MODEL = model
    _RMBG_TRANSFORM = transform_image
    return _RMBG_MODEL, _RMBG_TRANSFORM


def _remove_background_rgba_rmbg2(img: Image.Image) -> Image.Image:
    """Remove background using RMBG-2.0 to produce an RGBA PNG."""
    if img.mode in ("RGBA", "LA"):
        return img.convert("RGBA")

    model, transform_image = _load_rmbg()
    device = next(model.parameters()).device

    rgb = img.convert("RGB")
    input_images = transform_image(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().detach().cpu()
    pred = preds[0].squeeze(0)

    mask_pil = transforms.ToPILImage()(pred)
    mask_pil = mask_pil.resize(rgb.size)

    out = rgb.copy()
    out.putalpha(mask_pil)
    return out


def _remove_background_rgba_opencv(img: Image.Image) -> Image.Image:
    """
    Remove background to RGBA using OpenCV flood-fill (connected region) masks.

    Seeds:
    - four corners (background seeds)
    - center point (background seed)

    Strategy:
    - flood-fill from (corners + center) to get background-connected mask
    - alpha = inverse(background)
    """
    if img.mode in ("RGBA", "LA"):
        return img.convert("RGBA")

    import cv2  # type: ignore

    rgb = img.convert("RGB")
    arr = np.array(rgb)
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        return rgb.convert("RGBA")

    # OpenCV floodFill expects BGR.
    bgr = arr[:, :, ::-1].copy()

    tol = int(os.getenv("BG_REMOVE_TOL", "18"))
    lo = (tol, tol, tol)
    hi = (tol, tol, tol)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)  # 4-connectivity; write 255 into mask

    def _flood_mask(seeds):
        # Use a fresh copy for deterministic results.
        bgr_work = bgr.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for sx, sy in seeds:
            # clamp
            sx = max(0, min(w - 1, int(sx)))
            sy = max(0, min(h - 1, int(sy)))
            cv2.floodFill(bgr_work, mask, (sx, sy), 0, loDiff=lo, upDiff=hi, flags=flags)
        return mask[1 : h + 1, 1 : w + 1]

    corner_seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    cx, cy = (w // 2, h // 2)
    center_seed = [(cx, cy)]

    # Treat corners + center as background seeds (explicit requirement).
    bg_mask = _flood_mask(corner_seeds + center_seed)  # 255 where background-like regions connected to seeds
    bg = (bg_mask == 255).astype(np.uint8)

    # Foreground is everything not connected to background.
    alpha = (1 - bg) * 255

    # If result is degenerate (e.g. tol too large floods everything), retry with smaller tolerance.
    fg_ratio = float((alpha > 0).sum()) / float(h * w)
    if fg_ratio < 0.01:
        tol2 = max(3, tol // 2)
        lo2 = (tol2, tol2, tol2)
        hi2 = (tol2, tol2, tol2)
        flags2 = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        def _flood_mask2(seeds):
            bgr_work = bgr.copy()
            mask = np.zeros((h + 2, w + 2), np.uint8)
            for sx, sy in seeds:
                sx = max(0, min(w - 1, int(sx)))
                sy = max(0, min(h - 1, int(sy)))
                cv2.floodFill(bgr_work, mask, (sx, sy), 0, loDiff=lo2, upDiff=hi2, flags=flags2)
            return mask[1 : h + 1, 1 : w + 1]
        bg2 = (_flood_mask2(corner_seeds + center_seed) == 255).astype(np.uint8)
        alpha2 = (1 - bg2) * 255
        if float((alpha2 > 0).sum()) / float(h * w) > fg_ratio:
            alpha = alpha2

    # Soften edges slightly
    alpha = cv2.GaussianBlur(alpha.astype(np.uint8), (0, 0), sigmaX=1.0)

    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _save_image(img: Image.Image, out_dir: Path, stem: str) -> str:
    path = out_dir / f"{stem}.png"
    img.save(path)
    return str(path.resolve())


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
        "rmbg_device": _resolve_device(RMBG_DEVICE),
        "zimage_loaded": _ZIMAGE_PIPE is not None,
        "qwen_edit_loaded": _QWEN_EDIT_PIPE is not None,
        "rmbg_loaded": _RMBG_MODEL is not None,
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
    return {"path": _save_image(img, out_dir, stem), "engine": req.engine}


@app.post("/v1/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    num_inference_steps: int = Form(40),
    guidance_scale: float = Form(1.0),
    true_cfg_scale: float = Form(4.0),
    num_images_per_prompt: int = Form(1),
    seed: int = Form(0),
    out_dir: str = Form(""),
    remove_bg: bool = Form(True),
    remove_bg_method: str = Form("opencv"),  # opencv | rmbg2
    engine: str = Form("qwen_edit"),
) -> Dict[str, Any]:
    if engine != "qwen_edit":
        raise HTTPException(status_code=400, detail=f"unsupported engine for edit: {engine}")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    content = await image.read()
    try:
        src = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    out_dir_path = _ensure_out_dir(out_dir or None)
    pipe = _load_qwen_edit()

    with torch.inference_mode():
        out = pipe(
            image=[src],
            prompt=prompt,
            negative_prompt=QWEN_EDIT_NEGATIVE_PROMPT,
            generator=torch.manual_seed(int(seed)),
            true_cfg_scale=float(true_cfg_scale),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=int(num_images_per_prompt),
        )
    edited = out.images[0]

    stem_base = f"edit_{_now_tag()}_{uuid.uuid4().hex[:8]}"
    raw_path = _save_image(edited, out_dir_path, stem_base + "_raw")

    final_path = raw_path
    if remove_bg:
        if remove_bg_method == "rmbg2":
            rgba = _remove_background_rgba_rmbg2(edited)
        else:
            rgba = _remove_background_rgba_opencv(edited)
        final_path = _save_image(rgba, out_dir_path, stem_base + "_rgba")

    return {
        "path": final_path,
        "raw_path": raw_path,
        "engine": engine,
        "remove_bg": remove_bg,
        "remove_bg_method": remove_bg_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9106)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

