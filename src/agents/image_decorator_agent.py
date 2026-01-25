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
        self.poster_bg_prompt_tpl = load_prompt("config/prompts/poster_background_generate.txt")
        self.poster_bg_edit_prompt_tpl = load_prompt("config/prompts/poster_background_edit.txt")

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
        theme_character = str(self.render_cfg.get("theme_character", "cute research robot mascot"))
        poster_bg_enabled = bool(self.render_cfg.get("poster_background_enabled", False))
        poster_bg_edit_enabled = bool(self.render_cfg.get("poster_background_edit_enabled", True))
        poster_bg_remove_bg = bool(self.render_cfg.get("poster_background_remove_bg", True))
        poster_bg_long_edge = int(self.render_cfg.get("poster_bg_long_edge", 1536))

        out_dir = Path(state["output_dir"]) / "assets" / "decorative_backgrounds"
        out_dir.mkdir(parents=True, exist_ok=True)

        theme_color = "#1E3A8A"
        theme_color = (state.get("color_scheme") or {}).get("theme", theme_color)  

        log_agent_info(self.name, f"decorative backgrounds enabled; server={server_url}, out_dir={out_dir}")

        # 1) Select "low-usage" section containers by utilization ratio (used_area / container_area)
        styled_layout = state.get("styled_layout") or []
        section_containers = [
            e for e in styled_layout
            if isinstance(e, dict) and e.get("type") == "section_container" and e.get("section_id")
        ]
        if not section_containers:
            log_agent_warning(self.name, "no section_container found in styled_layout; skipping assignment")
            state["decorative_backgrounds"] = {"enabled": True, "sections": {}}
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

        # Build section_id -> section_title map from story_board (more reliable than rendered text).
        sid_to_title: Dict[str, str] = {}
        sb = state.get("story_board") or {}
        scp = sb.get("spatial_content_plan") if isinstance(sb, dict) else None
        sections_sb = scp.get("sections") if isinstance(scp, dict) else None
        if isinstance(sections_sb, list):
            for s in sections_sb:
                if isinstance(s, dict) and s.get("section_id"):
                    sid_to_title[str(s["section_id"])] = str(s.get("section_title", "")).strip()

        # Use up to 4 titles to drive a real 2×2 four-panel generation.
        panel_sids = chosen_sids[:4] + ([""] * max(0, 4 - len(chosen_sids[:4])))
        panel_titles = [
            sid_to_title.get(panel_sids[0], "Research Overview"),
            sid_to_title.get(panel_sids[1], "Method"),
            sid_to_title.get(panel_sids[2], "Results"),
            sid_to_title.get(panel_sids[3], "Analysis"),
        ]

        # 2) Build prompts for TRUE 2×2 sheet
        gen_prompt = Template(self.generate_prompt_tpl).render(
            theme_color=theme_color,
            theme_character=theme_character,
            panel_1_title=panel_titles[0],
            panel_2_title=panel_titles[1],
            panel_3_title=panel_titles[2],
            panel_4_title=panel_titles[3],
        ).strip()
        edit_prompt = Template(self.edit_prompt_tpl).render(
            theme_color=theme_color,
            theme_character=theme_character,
        ).strip()

        # 3) Call local server: generate base 2×2 sheet
        sheet_base_path = self._call_generate(
            server_url=server_url,
            prompt=gen_prompt,
            size=icon_size,
            out_dir=str(out_dir),
        )

        # 4) Call local server: edit + remove background (RGBA) on the whole sheet
        sheet_rgba_path = self._call_edit(
            server_url=server_url,
            image_path=sheet_base_path,
            prompt=edit_prompt,
            out_dir=str(out_dir),
            remove_bg=True,
        )

        sheet_rgba_path = self._ensure_local_copy(sheet_rgba_path, out_dir / "sheet_rgba.png")

        # 5) Split into 4 quadrants (each should be an independent panel)
        quadrant_paths = self._split_center_quadrants(sheet_rgba_path, out_dir)

        # 6) Assign one quadrant to each chosen section_id
        by_section: Dict[str, str] = {}
        if quadrant_strategy == "random":
            for sid in chosen_sids:
                by_section[str(sid)] = random.choice(quadrant_paths)
        else:
            # Prefer aligning first four sections to their panel positions for coherence.
            for i, sid in enumerate(chosen_sids):
                by_section[str(sid)] = quadrant_paths[i % len(quadrant_paths)]

        # 7) Generate full-poster background (second call)
        poster_bg_path = None
        if poster_bg_enabled:
            poster_title = ""
            nc = state.get("narrative_content") or {}
            if isinstance(nc, dict):
                meta = nc.get("meta") if isinstance(nc.get("meta"), dict) else {}
                if isinstance(meta, dict):
                    poster_title = str(meta.get("poster_title", "")).strip()
            if not poster_title:
                poster_title = str(state.get("poster_name", "Poster")).strip()

            pw = float(state.get("poster_width", 54))
            ph = float(state.get("poster_height", 36))
            # Keep ratio; set long edge pixels.
            if pw >= ph:
                w_px = int(poster_bg_long_edge)
                h_px = max(256, int(w_px * (ph / pw)))
            else:
                h_px = int(poster_bg_long_edge)
                w_px = max(256, int(h_px * (pw / ph)))

            bg_prompt = Template(self.poster_bg_prompt_tpl).render(
                poster_title=poster_title,
                theme_color=theme_color,
                theme_character=theme_character,
            ).strip()

            poster_bg_path = self._call_generate_rect(
                server_url=server_url,
                prompt=bg_prompt,
                width=w_px,
                height=h_px,
                out_dir=str(out_dir),
            )

            # Optional second-stage edit to make background more subtle (no background removal).
            if poster_bg_edit_enabled:
                bg_edit_prompt = Template(self.poster_bg_edit_prompt_tpl).render(
                    theme_color=theme_color,
                    theme_character=theme_character,
                ).strip()
                poster_bg_path = self._call_edit(
                    server_url=server_url,
                    image_path=poster_bg_path,
                    prompt=bg_edit_prompt,
                    out_dir=str(out_dir),
                    remove_bg=poster_bg_remove_bg,
                )

            poster_bg_path = self._ensure_local_copy(poster_bg_path, out_dir / "poster_background.png")

        state["decorative_backgrounds"] = {
            "enabled": True,
            "server_url": server_url,
            "sheet_base": sheet_base_path,
            "sheet_rgba": sheet_rgba_path,
            "quadrants": quadrant_paths,
            "sections": by_section,
            "poster_background": poster_bg_path,
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

    def _call_generate_rect(self, server_url: str, prompt: str, width: int, height: int, out_dir: str) -> str:
        url = f"{server_url}/v1/generate"
        payload = {
            "prompt": prompt,
            "height": int(height),
            "width": int(width),
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "seed": 123,
            "out_dir": out_dir,
            "engine": "zimage",
        }
        r = requests.post(url, json=payload, timeout=900)
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

