"""
pdf text and asset extraction
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from jinja2 import Template
from PIL import Image  # type: ignore

from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt_by_column_count
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.config.poster_config import load_config


class Parser:
    def __init__(self):
        self.name = "parser"
        # NOTE:
        # PosterGen3 originally used `marker` for PDF extraction.
        # We now migrate to MinerU (see `Paper2Slides/dev/mineru_pdf_extract_demo.py`)
        # while keeping downstream interfaces (raw_text/figures/tables) consistent.
        self.config = load_config()
        self.column_count = int(self.config.get("layout", {}).get("column_count", 3))

        self.clean_pattern = re.compile(r"<!--[\s\S]*?-->")
        self.enhanced_abt_prompt = load_prompt_by_column_count("narrative_abt_extraction.txt", self.column_count)
        self.visual_classification_prompt = load_prompt_by_column_count("classify_visuals.txt", self.column_count)
        self.title_authors_prompt = load_prompt_by_column_count("extract_title_authors.txt", self.column_count)
        self.section_extraction_prompt = load_prompt_by_column_count("extract_structured_sections.txt", self.column_count)
    
    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "starting foundation building")

        output_dir = Path(state["output_dir"])
        content_dir = output_dir / "content"
        assets_dir = output_dir / "assets"
        content_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        # extract raw text and assets
        raw_text, raw_result = self._extract_raw_text(state["pdf_path"], content_dir)
        figures, tables = self._extract_assets(raw_result, state["poster_name"], assets_dir)

        # extract title and authors from raw text
        title, authors = self._extract_title_authors(raw_text, state["text_model"])

        # generate narrative content
        narrative_content, inp_tok, out_tok = self._generate_narrative_content(raw_text, state["text_model"])
        state["tokens"].add_text(inp_tok, out_tok)

        # classify visual assets by importance
        classified_visuals, inp_tok2, out_tok2 = self._classify_visual_assets(figures, tables, raw_text, state["text_model"])
        state["tokens"].add_text(inp_tok2, out_tok2)

        # narrative metadata
        narrative_content["meta"] = {"poster_title": title, "authors": authors}

        # extract structured sections from raw text
        structured_sections = self._extract_structured_sections(raw_text, state["text_model"])

        # save artifacts and update state
        self._save_content(narrative_content, "narrative_content.json", content_dir)
        self._save_content(classified_visuals, "classified_visuals.json", content_dir)
        self._save_content(structured_sections, "structured_sections.json", content_dir)
        self._save_raw_text(raw_text, content_dir)

        state["raw_text"] = raw_text
        state["structured_sections"] = structured_sections
        state["narrative_content"] = narrative_content
        state["classified_visuals"] = classified_visuals
        state["images"] = figures
        state["tables"] = tables
        state["current_agent"] = self.name

        log_agent_success(self.name, f"extracted raw text, {len(figures)} images, and {len(tables)} tables")
        log_agent_success(self.name, f"extracted title: {title}")
        log_agent_success(self.name, "generated enhanced abt narrative")
        log_agent_success(
            self.name,
            f"classified visuals: key={classified_visuals.get('key_visual', 'none')}, "
            f"problem_ill={len(classified_visuals.get('problem_illustration', []))}, "
            f"method_wf={len(classified_visuals.get('method_workflow', []))}, "
            f"main_res={len(classified_visuals.get('main_results', []))}, "
            f"comp_res={len(classified_visuals.get('comparative_results', []))}, "
            f"support={len(classified_visuals.get('supporting', []))}",
        )
        
        return state
    
    # -------------------------
    # MinerU extraction helpers
    # -------------------------
    def _safe_image_size(self, path: Path) -> Tuple[int, int]:
        with Image.open(path) as im:
            return int(im.width), int(im.height)

    def _normalize_content_item_type(self, t: Any) -> str:
        return str(t or "").strip().lower()

    def _extract_from_content_list(
        self,
        content_list: List[Dict[str, Any]],
        out_dir: Path,
        assets_dir: Path,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Convert MinerU `content_list` to:
        - figures: {"1": {caption, path, width, height, aspect}, ...}
        - tables:  {"1": {caption, path, width, height, aspect}, ...}
        - caption_map: used to write `fig_tab_caption_mapping.json` for compatibility/debug.
        """
        figures: Dict[str, Any] = {}
        tables: Dict[str, Any] = {}
        caption_map: Dict[str, Any] = {}

        fig_idx = 0
        tab_idx = 0

        def _resolve_img_abs(rel_or_abs: str) -> Optional[Path]:
            if not rel_or_abs:
                return None
            p = Path(rel_or_abs)
            if p.is_absolute():
                return p
            cand = (out_dir / p).resolve()
            if cand.exists():
                return cand
            cand2 = (assets_dir / p.name).resolve()
            if cand2.exists():
                return cand2
            return cand

        for item in content_list:
            t = self._normalize_content_item_type(item.get("type"))
            img_path = item.get("img_path") or ""
            if not img_path:
                continue

            abs_img = _resolve_img_abs(str(img_path))
            if abs_img is None:
                continue

            assets_dir.mkdir(parents=True, exist_ok=True)
            if abs_img.exists():
                dst = (assets_dir / abs_img.name).resolve()
                if dst != abs_img.resolve():
                    shutil.copyfile(abs_img, dst)
                    abs_img = dst

            w, h = self._safe_image_size(abs_img)
            aspect = (w / h) if h else 1
            page_no = item.get("page_no") or item.get("page")

            if t == "image":
                fig_idx += 1
                caption_list = item.get("image_caption") or []
                caption = caption_list[0] if caption_list else f"Figure {fig_idx}"
                figures[str(fig_idx)] = {
                    "caption": caption,
                    "path": str(abs_img),
                    "width": w,
                    "height": h,
                    "aspect": aspect,
                }
                caption_map[Path(img_path).name] = {
                    "block_type": "Figure",
                    "captions": caption_list,
                    "page": page_no,
                    "path": str(abs_img),
                }

            elif t == "table":
                tab_idx += 1
                caption_list = item.get("table_caption") or []
                caption = caption_list[0] if caption_list else f"Table {tab_idx}"
                tables[str(tab_idx)] = {
                    "caption": caption,
                    "path": str(abs_img),
                    "width": w,
                    "height": h,
                    "aspect": aspect,
                }
                caption_map[Path(img_path).name] = {
                    "block_type": "Table",
                    "captions": caption_list,
                    "page": page_no,
                    "path": str(abs_img),
                }

        return figures, tables, caption_map

    def _extract_with_mineru_cli(self, pdf_path: Path, out_dir: Path, content_dir: Path, assets_dir: Path, mineru_cli: str = "mineru") -> List[Dict[str, Any]]:
        """MinerU CLI backend. Returns content_list; writes raw.md and copies images."""
        mineru_out_dir = out_dir / "tmp_mineru_cli_output"
        mineru_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [mineru_cli, "-p", str(pdf_path), "-o", str(mineru_out_dir)]
        log_agent_info(self.name, "[mineru(cli)] running: " + " ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        pdf_stem = pdf_path.stem
        md_candidates = list(mineru_out_dir.rglob(f"{pdf_stem}.md")) or list(mineru_out_dir.rglob("*.md"))
        if not md_candidates:
            raise RuntimeError("未在 MinerU CLI 输出中找到 markdown 文件（*.md）")
        mineru_md = md_candidates[0]

        clist_candidates = list(mineru_out_dir.rglob(f"{pdf_stem}_content_list.json")) or list(mineru_out_dir.rglob("*_content_list.json"))
        if not clist_candidates:
            raise RuntimeError("未在 MinerU CLI 输出中找到 content_list.json（*_content_list.json）")
        content_list_path = clist_candidates[0]

        images_dir = mineru_out_dir / "images"
        if not images_dir.exists():
            imgs = [p for p in mineru_out_dir.rglob("images") if p.is_dir()]
            if imgs:
                images_dir = imgs[0]

        md_text = mineru_md.read_text(encoding="utf-8")
        (content_dir / "raw.md").write_text(md_text, encoding="utf-8")

        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
        if not isinstance(content_list, list):
            raise RuntimeError(f"content_list.json 非 list：{type(content_list)}")

        # best-effort copy images into assets_dir
        for item in content_list:
            rel_img = item.get("img_path")
            if not rel_img:
                continue
            src_path = (mineru_out_dir / rel_img).resolve()
            if not src_path.exists() and images_dir.exists():
                src_path2 = images_dir / Path(rel_img).name
                if src_path2.exists():
                    src_path = src_path2.resolve()
            if not src_path.exists():
                continue
            dst_path = (assets_dir / src_path.name).resolve()
            if dst_path != src_path:
                shutil.copyfile(src_path, dst_path)

        return content_list

    def _extract_raw_text(self, pdf_path: str, content_dir: Path) -> Tuple[str, Any]:
        """
        MinerU-based extraction.
        Returns:
          - raw markdown string
          - raw_result: {"content_list": [...], "out_dir": str, "assets_dir": str}
        """
        log_agent_info(self.name, "converting pdf to raw text (MinerU)")
        pdf_p = Path(pdf_path).expanduser().resolve()
        out_dir = content_dir.parent
        assets_dir = out_dir / "assets"
        content_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Use MinerU CLI only (simpler, stable behavior).
        content_list = self._extract_with_mineru_cli(pdf_p, out_dir, content_dir, assets_dir, mineru_cli="mineru")
        log_agent_info(self.name, "[mineru] backend=cli")

        text = (content_dir / "raw.md").read_text(encoding="utf-8")
        text = self.clean_pattern.sub("", text)
        (content_dir / "raw.md").write_text(text, encoding="utf-8")

        log_agent_info(self.name, f"extracted {len(text)} chars")
        raw_result = {
            "content_list": content_list,
            "out_dir": str(out_dir),
            "assets_dir": str(assets_dir),
        }
        return text, raw_result

    def _generate_narrative_content(self, text: str, config) -> Tuple[Dict, int, int]:
        log_agent_info(self.name, "generating abt narrative")
        agent = LangGraphAgent("expert poster design consultant", config)

        prompt = Template(self.enhanced_abt_prompt).render(markdown_document=text)
        agent.reset()
        response = agent.step(prompt)
        narrative = extract_json(response.content)
        if not ("and" in narrative and "but" in narrative and "therefore" in narrative):
            raise ValueError("invalid narrative format: missing and/but/therefore")
        return narrative, response.input_tokens, response.output_tokens
    
    def _save_content(self, content: Dict, filename: str, content_dir: Path):
        with open(content_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
    
    def _save_raw_text(self, raw_text: str, content_dir: Path):
        with open(content_dir / "raw.md", 'w', encoding='utf-8') as f:
            f.write(raw_text)
    
    def _extract_assets(self, result, name: str, assets_dir: Path) -> Tuple[Dict, Dict]:
        """Extract figures/tables from MinerU content_list. Interface-compatible."""
        log_agent_info(self.name, "extracting assets (MinerU)")

        if not isinstance(result, dict) or "content_list" not in result:
            raise ValueError("invalid mineru raw_result: missing content_list")

        out_dir = Path(result.get("out_dir", assets_dir.parent))
        content_list = result["content_list"]
        if not isinstance(content_list, list):
            raise ValueError(f"invalid mineru content_list: {type(content_list)}")

        figures, tables, caption_map = self._extract_from_content_list(
            content_list=content_list,
            out_dir=out_dir,
            assets_dir=assets_dir,
        )

        # Persist for debugging / compatibility with existing downstream artifacts.
        with open(assets_dir / "figures.json", "w", encoding="utf-8") as f:
            json.dump(figures, f, indent=2, ensure_ascii=False)
        with open(assets_dir / "tables.json", "w", encoding="utf-8") as f:
            json.dump(tables, f, indent=2, ensure_ascii=False)
        with open(assets_dir / "fig_tab_caption_mapping.json", "w", encoding="utf-8") as f:
            json.dump(caption_map, f, indent=2, ensure_ascii=False)

        return figures, tables

    def _extract_title_authors(self, text: str, config) -> Tuple[str, str]:
        """extract title and authors via llm api"""
        log_agent_info(self.name, "extracting title and authors with llm")
        agent = LangGraphAgent("expert academic paper parser", config)

        prompt = Template(self.title_authors_prompt).render(markdown_document=text)
        agent.reset()
        response = agent.step(prompt)
        result = extract_json(response.content)
        title = str(result.get("title", "")).strip()
        authors = str(result.get("authors", "")).strip()
        if not (title and authors):
            raise ValueError("failed to extract title/authors from LLM response")
        return title, authors
    
    
    def _classify_visual_assets(self, figures: Dict, tables: Dict, raw_text: str, config) -> Tuple[Dict, int, int]:
        # combine all visuals for classification
        all_visuals = []
        for fig_id, fig_data in figures.items():
            all_visuals.append({
                "id": f"figure_{fig_id}",
                "type": "figure", 
                "caption": fig_data.get("caption", ""),
                "aspect_ratio": fig_data.get("aspect", 1.0)
            })
        
        for tab_id, tab_data in tables.items():
            all_visuals.append({
                "id": f"table_{tab_id}",
                "type": "table",
                "caption": tab_data.get("caption", ""),
                "aspect_ratio": tab_data.get("aspect", 1.0)
            })
        
        if not all_visuals:
            return {"key_visual": None, "problem_illustration": [], "method_workflow": [], "main_results": [], "comparative_results": [], "supporting": []}, 0, 0
            
        log_agent_info(self.name, f"classifying {len(all_visuals)} visual assets")
        agent = LangGraphAgent("expert poster designer", config)

        prompt = Template(self.visual_classification_prompt).render(visuals_list=json.dumps(all_visuals, indent=2))
        agent.reset()
        response = agent.step(prompt)
        classification = extract_json(response.content)

        required_keys = ["key_visual", "problem_illustration", "method_workflow", "main_results", "comparative_results", "supporting"]
        if not all(key in classification for key in required_keys):
            raise ValueError(f"invalid visual classification response: missing keys in {required_keys}")
        return classification, response.input_tokens, response.output_tokens

    def _extract_structured_sections(self, raw_text: str, config) -> Dict:
        """extract structured sections from raw paper text"""
        
        log_agent_info(self.name, "extracting structured sections from paper")
        agent = LangGraphAgent("expert paper section extractor", config)

        prompt = Template(self.section_extraction_prompt).render(raw_text=raw_text)
        agent.reset()
        response = agent.step(prompt)
        structured_sections = extract_json(response.content)
        if not self._validate_structured_sections(structured_sections):
            raise ValueError("invalid structured sections format")
        log_agent_success(self.name, f"extracted {len(structured_sections.get('paper_sections', []))} structured sections")
        return structured_sections
    
    def _validate_structured_sections(self, structured_sections: Dict) -> bool:
        """validate structured sections format"""
        if "paper_sections" not in structured_sections:
            log_agent_warning(self.name, "validation error: missing 'paper_sections'")
            return False
        
        sections = structured_sections["paper_sections"]
        if not isinstance(sections, list) or len(sections) < 3:
            log_agent_warning(self.name, f"validation error: need at least 3 sections, got {len(sections)}")
            return False
        
        # validate each section
        for i, section in enumerate(sections):
            required_fields = ["section_name", "section_type", "content"]
            for field in required_fields:
                if field not in section:
                    log_agent_warning(self.name, f"validation error: section {i} missing '{field}'")
                    return False
        
        return True


def parser_node(state: PosterState) -> PosterState:
    return Parser()(state) 