"""
pdf text and asset extraction
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Any, Tuple

from marker.converters.pdf import PdfConverter
from marker.renderers.markdown import MarkdownRenderer
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.schema import BlockTypes
from jinja2 import Template

from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.config.poster_config import load_config


class Parser:
    def __init__(self):
        self.name = "parser"
        config_data = load_config()
        batch_config = config_data["pdf_processing"]["batch_sizes"]
        config = {
            "recognition_batch_size": batch_config["recognition"],
            "layout_batch_size": batch_config["layout"],
            "detection_batch_size": batch_config["detection"], 
            "table_rec_batch_size": batch_config["table_rec"],
            "ocr_error_batch_size": batch_config["ocr_error"],
            "equation_batch_size": batch_config["equation"],
            "disable_tqdm": False,
        }
        
        self.converter = PdfConverter(artifact_dict=create_model_dict(), config=config)
        self.clean_pattern = re.compile(r"<!--[\s\S]*?-->")
        self.enhanced_abt_prompt = load_prompt("config/prompts/narrative_abt_extraction.txt")
        self.visual_classification_prompt = load_prompt("config/prompts/classify_visuals.txt")
        self.title_authors_prompt = load_prompt("config/prompts/extract_title_authors.txt")
        self.section_extraction_prompt = load_prompt("config/prompts/extract_structured_sections.txt")
    
    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "starting foundation building")
        
        try:
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
            narrative_content["meta"] = {
                "poster_title": title,
                "authors": authors
            }
            
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
            log_agent_success(self.name, f"classified visuals: key={classified_visuals.get('key_visual', 'none')}, problem_ill={len(classified_visuals.get('problem_illustration', []))}, method_wf={len(classified_visuals.get('method_workflow', []))}, main_res={len(classified_visuals.get('main_results', []))}, comp_res={len(classified_visuals.get('comparative_results', []))}, support={len(classified_visuals.get('supporting', []))}")
            
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(str(e))
        
        return state
    
    def _extract_raw_text(self, pdf_path: str, content_dir: Path) -> Tuple[str, Any]:
        log_agent_info(self.name, "converting pdf to raw text")
        document = self.converter.build_document(pdf_path)
        
        # create renderer and get rendered output from the existing document
        renderer = self.converter.resolve_dependencies(MarkdownRenderer)
        rendered = renderer(document)
        
        text, _, images = text_from_rendered(rendered)
        text = self.clean_pattern.sub("", text)
        
        (content_dir / "raw.md").write_text(text, encoding="utf-8")
        
        log_agent_info(self.name, f"extracted {len(text)} chars")
        
        raw_result = (document, rendered, images)
        return text, raw_result

    def _generate_narrative_content(self, text: str, config) -> Tuple[Dict, int, int]:
        log_agent_info(self.name, "generating abt narrative")
        agent = LangGraphAgent("expert poster design consultant", config)
        
        for attempt in range(3):
            try:
                prompt = Template(self.enhanced_abt_prompt).render(markdown_document=text)
                agent.reset()
                response = agent.step(prompt)
                
                narrative = extract_json(response.content)
                
                if "and" in narrative and "but" in narrative and "therefore" in narrative:
                    return narrative, response.input_tokens, response.output_tokens

            except Exception as e:
                log_agent_warning(self.name, f"attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise

        raise ValueError("failed to generate enhanced narrative after 3 attempts")
    
    def _save_content(self, content: Dict, filename: str, content_dir: Path):
        with open(content_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
    
    def _save_raw_text(self, raw_text: str, content_dir: Path):
        with open(content_dir / "raw.md", 'w', encoding='utf-8') as f:
            f.write(raw_text)
    
    def _extract_assets(self, result, name: str, assets_dir: Path) -> Tuple[Dict, Dict]:
        log_agent_info(self.name, "extracting assets")
        
        document, rendered, marker_images = result
        
        caption_map = self._extract_captions(document)
        
        figures = {}
        tables = {}
        image_count = 0
        table_count = 0
        
        for img_name, pil_image in marker_images.items():
            caption_info = caption_map.get(img_name, {'captions': [], 'block_type': 'Unknown'})
            
            if 'table' in img_name.lower() or 'Table' in img_name or caption_info.get('block_type') == 'Table':
                table_count += 1
                path = assets_dir / f"table-{table_count}.png"
                pil_image.save(path, "PNG")
                
                tables[str(table_count)] = {
                    'caption': caption_info['captions'][0] if caption_info['captions'] else f"Table {table_count}",
                    'path': str(path),
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'aspect': pil_image.width / pil_image.height if pil_image.height > 0 else 1,
                }
            else:
                image_count += 1
                path = assets_dir / f"figure-{image_count}.png"
                pil_image.save(path, "PNG")
                
                figures[str(image_count)] = {
                    'caption': caption_info['captions'][0] if caption_info['captions'] else f"Figure {image_count}",
                    'path': str(path),
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'aspect': pil_image.width / pil_image.height if pil_image.height > 0 else 1,
                }
        
        with open(assets_dir / "figures.json", 'w', encoding='utf-8') as f:
            json.dump(figures, f, indent=2)
        with open(assets_dir / "tables.json", 'w', encoding='utf-8') as f:
            json.dump(tables, f, indent=2)
        with open(assets_dir / "fig_tab_caption_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(caption_map, f, indent=2, ensure_ascii=False)
        
        return figures, tables

    def _extract_captions(self, document):
        caption_map = {}
        
        for page in document.pages:
            for block_id in page.structure:
                block = page.get_block(block_id)
                
                if block.block_type in [BlockTypes.FigureGroup, BlockTypes.TableGroup, BlockTypes.PictureGroup]:
                    child_blocks = block.structure_blocks(page)
                    figure_or_table = None
                    captions = []
                    
                    for child in child_blocks:
                        child_block = page.get_block(child)
                        if child_block.block_type in [BlockTypes.Figure, BlockTypes.Table, BlockTypes.Picture]:
                            figure_or_table = child_block
                        elif child_block.block_type in [BlockTypes.Caption, BlockTypes.Footnote]:
                            captions.append(child_block.raw_text(document))
                    
                    if figure_or_table:
                        image_filename = f"{figure_or_table.id.to_path()}.jpeg"
                        caption_map[image_filename] = {
                            'block_id': str(figure_or_table.id),
                            'block_type': str(figure_or_table.block_type),
                            'captions': captions,
                            'page': page.page_id
                        }
                
                elif block.block_type in [BlockTypes.Figure, BlockTypes.Table, BlockTypes.Picture]:
                    image_filename = f"{block.id.to_path()}.jpeg"
                    if image_filename not in caption_map:
                        nearby_captions = self._find_nearby_captions(page, block, document)
                        caption_map[image_filename] = {
                            'block_id': str(block.id),
                            'block_type': str(block.block_type),
                            'captions': nearby_captions,
                            'page': page.page_id
                        }
        
        return caption_map

    def _find_nearby_captions(self, page, target_block, document):
        captions = []
        
        # Check all blocks on the page for captions
        for block_id in page.structure:
            block = page.get_block(block_id)
            if block.block_type in [BlockTypes.Caption, BlockTypes.Text]:
                caption_text = block.raw_text(document)
                # Look for figure/table keywords and check if it's nearby
                if any(keyword in caption_text for keyword in ['Figure', 'Table', 'Fig.']):
                    captions.append(caption_text)
        
        # If no captions found, try previous/next blocks
        if not captions:
            for block in [page.get_prev_block(target_block), page.get_next_block(target_block)]:
                if block and block.block_type in [BlockTypes.Caption, BlockTypes.Text]:
                    caption_text = block.raw_text(document)
                    if any(keyword in caption_text for keyword in ['Figure', 'Table', 'Fig.']):
                        captions.append(caption_text)
        
        return captions

    def _cleanup_unused_assets(self, output_dir: Path, name: str, images: Dict, tables: Dict):
        valid_paths = set()
        for img_data in images.values():
            valid_paths.add(Path(img_data['path']).name)
        for table_data in tables.values():
            valid_paths.add(Path(table_data['path']).name)
        
        for png_file in output_dir.glob(f"{name}-*.png"):
            if png_file.name not in valid_paths:
                png_file.unlink()

    def _extract_title_authors(self, text: str, config) -> Tuple[str, str]:
        """extract title and authors via llm api"""
        log_agent_info(self.name, "extracting title and authors with llm")
        agent = LangGraphAgent("expert academic paper parser", config)
        
        for attempt in range(3):
            try:
                prompt = Template(self.title_authors_prompt).render(markdown_document=text)
                agent.reset()
                response = agent.step(prompt)
                
                result = extract_json(response.content)

                if "title" in result and "authors" in result:
                    title = result["title"].strip()
                    authors = result["authors"].strip()
                    
                    # validate format
                    if title and authors:
                        return title, authors
                        
            except Exception as e:
                log_agent_warning(self.name, f"title/authors extraction attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    return "Untitled", "Authors not found"
        
        return "Untitled", "Authors not found"
    
    
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
        
        for attempt in range(3):
            try:
                prompt = Template(self.visual_classification_prompt).render(
                    visuals_list=json.dumps(all_visuals, indent=2)
                )
                
                agent.reset()
                response = agent.step(prompt)
                classification = extract_json(response.content)
                
                # validate classification
                required_keys = ["key_visual", "problem_illustration", "method_workflow", "main_results", "comparative_results", "supporting"]
                if all(key in classification for key in required_keys):
                    return classification, response.input_tokens, response.output_tokens
                    
            except Exception as e:
                log_agent_warning(self.name, f"visual classification attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    # fallback classification
                    return self._fallback_visual_classification(all_visuals), 0, 0
        
        return self._fallback_visual_classification(all_visuals), 0, 0
    
    def _fallback_visual_classification(self, visuals):
        # simple rule-based fallback
        classification = {"key_visual": None, "main_results": [], "method_diagrams": [], "supporting": []}
        
        for visual in visuals:
            caption = visual.get("caption", "").lower()
            if "result" in caption or "performance" in caption or "comparison" in caption:
                classification["main_results"].append(visual["id"])
            elif "method" in caption or "architecture" in caption or "framework" in caption:
                classification["method_diagrams"].append(visual["id"])
            else:
                classification["supporting"].append(visual["id"])
        
        # select key visual from main results or method diagrams
        if classification["main_results"]:
            classification["key_visual"] = classification["main_results"][0]
        elif classification["method_diagrams"]:
            classification["key_visual"] = classification["method_diagrams"][0]
        
        return classification

    def _extract_structured_sections(self, raw_text: str, config) -> Dict:
        """extract structured sections from raw paper text"""
        
        log_agent_info(self.name, "extracting structured sections from paper")
        agent = LangGraphAgent("expert paper section extractor", config)
        
        for attempt in range(3):
            try:
                prompt = Template(self.section_extraction_prompt).render(raw_text=raw_text)
                agent.reset()
                response = agent.step(prompt)
                
                structured_sections = extract_json(response.content)
                
                if self._validate_structured_sections(structured_sections):
                    log_agent_success(self.name, f"extracted {len(structured_sections.get('paper_sections', []))} structured sections")
                    return structured_sections
                else:
                    log_agent_warning(self.name, f"attempt {attempt + 1}: invalid structured sections")
                    
            except Exception as e:
                log_agent_warning(self.name, f"section extraction attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise ValueError("failed to extract structured sections after multiple attempts")

        # fallback empty structure
        return {
            "paper_sections": [],
            "paper_structure": {
                "total_sections": 0,
                "foundation_sections": 0,
                "method_sections": 0,
                "evaluation_sections": 0,
                "conclusion_sections": 0
            }
        }
    
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