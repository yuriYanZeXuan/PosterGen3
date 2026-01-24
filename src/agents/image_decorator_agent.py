"""
Image decorator agent

Calls the local FastAPI image server in `dev/image_server.py` to:
- generate a base icon image
- edit it into a cartoon sticker and remove background (RGBA)
- split the RGBA image into 4 center quadrants
- assign one quadrant as subtle background to a few "low-usage" section containers
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from PIL import Image
from jinja2 import Template

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.langgraph_utils import load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_warning


class ImageDecoratorAgent:
    def __init__(self):
        self.name = "image_decorator_agent"
        self.config = load_config()
        self.render_cfg = (self.config.get("rendering", {}) or {}).get("decorative_backgrounds", {}) or {}
        self.generate_prompt_tpl = load_prompt("config/prompts/image_decorator_generate.txt")
        self.edit_prompt_tpl = load_prompt("config/prompts/image_decorator_edit.txt")

    def __call__(self, state: PosterState) -> PosterState:
        enabled = bool(self.render_cfg.get("enabled", False))
        if not enabled:
            state["decorative_backgrounds"] = {"enabled": False}
            state["current_agent"] = self.name
            return state

        server_url = str(self.render_cfg.get("server_url", "http://127.0.0.1:9106")).rstrip("/")
        num_sections = int(self.render_cfg.get("num_sections", 3))
        quadrant_strategy = str(self.render_cfg.get("quadrant_strategy", "cycle"))
        icon_size = int(self.render_cfg.get("icon_size", 768))

        out_dir = Path(state["output_dir"]) / "assets" / "decorative_backgrounds"
        out_dir.mkdir(parents=True, exist_ok=True)

        theme_color = "#1E3A8A"
        try:
            theme_color = (state.get("color_scheme") or {}).get("theme", theme_color)  # type: ignore[assignment]
        except Exception:
            pass

        log_agent_info(self.name, f"decorative backgrounds enabled; server={server_url}, out_dir={out_dir}")

        # 1) Build prompts
        gen_prompt = Template(self.generate_prompt_tpl).render(theme_color=theme_color).strip()
        edit_prompt = Template(self.edit_prompt_tpl).render(theme_color=theme_color).strip()

        # 2) Call local server: generate base image
        base_img_path = self._call_generate(
            server_url=server_url,
            prompt=gen_prompt,
            size=icon_size,
            out_dir=str(out_dir),
        )

        # 3) Call local server: edit + remove background (RGBA)
        rgba_img_path = self._call_edit(
            server_url=server_url,
            image_path=base_img_path,
            prompt=edit_prompt,
            out_dir=str(out_dir),
            remove_bg=True,
        )

        # Ensure final RGBA is copied into our output_dir (some servers may save elsewhere).
        rgba_img_path = self._ensure_local_copy(rgba_img_path, out_dir / "icon_rgba.png")

        # 4) Split into 4 quadrants around center
        quadrant_paths = self._split_center_quadrants(rgba_img_path, out_dir)

        # 5) Select "low-usage" section containers by utilization ratio (used_area / container_area)
        styled_layout = state.get("styled_layout") or []
        section_containers = [
            e for e in styled_layout
            if isinstance(e, dict) and e.get("type") == "section_container" and e.get("section_id")
        ]
        if not section_containers:
            log_agent_warning(self.name, "no section_container found in styled_layout; skipping assignment")
            state["decorative_backgrounds"] = {"enabled": True, "quadrants": quadrant_paths, "sections": {}}
            state["current_agent"] = self.name
            return state

        # Build per-section utilization based on bbox intersection of content elements within each container.
        content_types = {
            "text",
            "visual",
            "mixed",
            "section_title",
            "title_accent_block",
            "title_accent_line",
            "title_accent_curve",
        }

        def _bbox(e: Dict[str, Any]) -> Tuple[float, float, float, float]:
            x0 = float(e.get("x", 0))
            y0 = float(e.get("y", 0))
            x1 = x0 + float(e.get("width", 0))
            y1 = y0 + float(e.get("height", 0))
            return x0, y0, x1, y1

        def _intersect_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            return (ix1 - ix0) * (iy1 - iy0)

        containers: Dict[str, Dict[str, Any]] = {}
        for c in section_containers:
            sid = str(c.get("section_id"))
            cb = _bbox(c)
            area = _intersect_area(cb, cb)
            if area <= 0:
                continue
            containers[sid] = {"bbox": cb, "area": area, "used": 0.0}

        # Accumulate used area for each container.
        for e in styled_layout:
            if not isinstance(e, dict):
                continue
            if e.get("type") not in content_types:
                continue
            try:
                eb = _bbox(e)
            except Exception:
                continue

            # Prefer id-based assignment if possible, otherwise use maximal intersection.
            assigned_sid = None
            eid = e.get("id")
            if isinstance(eid, str):
                for sid in containers.keys():
                    if eid.startswith(f"{sid}_") or eid == sid:
                        assigned_sid = sid
                        break

            if assigned_sid is None:
                best_sid = None
                best_area = 0.0
                for sid, meta in containers.items():
                    ia = _intersect_area(eb, meta["bbox"])
                    if ia > best_area:
                        best_area = ia
                        best_sid = sid
                if best_sid is not None and best_area > 0:
                    containers[best_sid]["used"] += best_area
            else:
                meta = containers.get(assigned_sid)
                if meta:
                    containers[assigned_sid]["used"] += _intersect_area(eb, meta["bbox"])

        def _utilization(sid: str) -> float:
            meta = containers[sid]
            return float(meta["used"]) / float(meta["area"]) if meta["area"] > 0 else 1.0

        # Sort by utilization ascending (lower usage first).
        ranked = sorted(containers.keys(), key=_utilization)
        chosen_sids = ranked[: max(0, min(num_sections, len(ranked)))]
        chosen = [c for c in section_containers if str(c.get("section_id")) in set(chosen_sids)]

        # 6) Assign one quadrant to each chosen section_id
        by_section: Dict[str, str] = {}
        if quadrant_strategy == "random":
            for e in chosen:
                sid = str(e["section_id"])
                by_section[sid] = random.choice(quadrant_paths)
        else:
            # cycle
            for i, e in enumerate(chosen):
                sid = str(e["section_id"])
                by_section[sid] = quadrant_paths[i % len(quadrant_paths)]

        state["decorative_backgrounds"] = {
            "enabled": True,
            "server_url": server_url,
            "icon_base": base_img_path,
            "icon_rgba": rgba_img_path,
            "quadrants": quadrant_paths,
            "sections": by_section,
        }
        state["current_agent"] = self.name
        log_agent_success(self.name, f"assigned decorative backgrounds to {len(by_section)} sections")
        return state

    def _call_generate(self, server_url: str, prompt: str, size: int, out_dir: str) -> str:
        url = f"{server_url}/v1/generate"
        payload = {
            "prompt": prompt,
            "height": int(size),
            "width": int(size),
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "seed": 42,
            "out_dir": out_dir,
            "engine": "zimage",
        }
        r = requests.post(url, json=payload, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"generate failed ({r.status_code}): {r.text}")
        data = r.json()
        return str(data["path"])

    def _call_edit(self, server_url: str, image_path: str, prompt: str, out_dir: str, remove_bg: bool) -> str:
        url = f"{server_url}/v1/edit"
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/png")}
            form = {
                "engine": "qwen_edit",
                "prompt": prompt,
                "negative_prompt": " ",
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "true_cfg_scale": 4.0,
                "num_images_per_prompt": 1,
                "seed": 0,
                "remove_bg": "true" if remove_bg else "false",
                "out_dir": out_dir,
            }
            r = requests.post(url, files=files, data=form, timeout=1800)
        if r.status_code != 200:
            raise RuntimeError(f"edit failed ({r.status_code}): {r.text}")
        data = r.json()
        return str(data["path"])

    def _ensure_local_copy(self, src_path: str, dst_path: Path) -> str:
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"decorator image not found: {src}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst_path.resolve():
            shutil.copyfile(src, dst_path)
        return str(dst_path.resolve())

    def _split_center_quadrants(self, rgba_path: str, out_dir: Path) -> List[str]:
        img = Image.open(rgba_path).convert("RGBA")
        w, h = img.size
        cx, cy = w // 2, h // 2
        boxes: List[Tuple[int, int, int, int]] = [
            (0, 0, cx, cy),      # top-left
            (cx, 0, w, cy),      # top-right
            (0, cy, cx, h),      # bottom-left
            (cx, cy, w, h),      # bottom-right
        ]
        paths: List[str] = []
        for i, box in enumerate(boxes):
            q = img.crop(box)
            p = out_dir / f"quadrant_{i}.png"
            q.save(p)
            paths.append(str(p.resolve()))
        return paths


def image_decorator_agent_node(state: PosterState) -> Dict[str, Any]:
    result = ImageDecoratorAgent()(state)
    # Keep langgraph node signature consistent with other agents.
    return {
        **state,
        "decorative_backgrounds": result.get("decorative_backgrounds"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors"),
        "tokens": result.get("tokens"),
    }

