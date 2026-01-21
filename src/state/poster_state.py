"""poster state management"""

from typing import Dict, Any, Optional, List, Tuple, TypedDict
from dataclasses import dataclass
import time


@dataclass
class ModelConfig:
    model_name: str
    provider: str
    temperature: float = 0.7
    max_tokens: int = 4096
    # Optional overrides to avoid relying on environment variables.
    # When set, `utils.langgraph_utils.create_model()` will prefer these values.
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass 
class TokenUsage:
    input_text: int = 0
    output_text: int = 0
    input_vision: int = 0
    output_vision: int = 0
    
    def add_text(self, inp: int, out: int):
        self.input_text += inp
        self.output_text += out
    
    def add_vision(self, inp: int, out: int):
        self.input_vision += inp
        self.output_vision += out


class PosterState(TypedDict):
    # core paths
    pdf_path: str
    output_dir: str
    poster_name: str
    
    # model configs
    text_model: ModelConfig
    vision_model: ModelConfig
    
    # processing results
    images: Optional[Dict[str, Any]]
    tables: Optional[Dict[str, Any]]
    narrative: Optional[Dict[str, str]]
    poster_plan: Optional[List[Dict[str, Any]]]
    poster_width: int
    poster_height: int
    wireframe_layout: Optional[List[Dict[str, Any]]]
    content_filled_layout: Optional[List[Dict[str, Any]]]
    final_layout: Optional[List[Dict[str, Any]]]
    
    narrative_content: Optional[Dict[str, Any]]
    classified_visuals: Optional[Dict[str, Any]]
    structured_sections: Optional[Dict[str, Any]]
    story_board: Optional[Dict[str, Any]]
    curated_content: Optional[Dict[str, Any]]
    design_layout: Optional[List[Dict[str, Any]]]
    section_title_design: Optional[Dict[str, Any]]
    color_scheme: Optional[Dict[str, str]]
    keywords: Optional[Dict[str, Any]]
    styled_layout: Optional[List[Dict[str, Any]]]
    
    # poster assets
    url: str
    logo_path: str
    aff_logo_path: Optional[str]
    
    # metadata
    tokens: TokenUsage
    current_agent: str
    errors: List[str]


def create_state(pdf_path: str, text_model: str = "gpt-4.1-2025-04-14", vision_model: str = "gpt-4.1-2025-04-14", width: int = 84, height: int = 42, url: str = "", logo_path: str = "", aff_logo_path: str = "") -> PosterState:
    """create initial poster state"""
    from pathlib import Path
    
    base_name = Path(pdf_path).parent.name or "test_poster"

    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting outputs across runs by appending an incrementing suffix.
    # Example: output/foo, output/foo_001, output/foo_002, ...
    candidate = output_root / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        poster_name = base_name
        output_dir = str(candidate)
    else:
        poster_name = base_name
        output_dir = str(candidate)
        for i in range(1, 10000):
            run_name = f"{base_name}_{i:03d}"
            run_dir = output_root / run_name
            if not run_dir.exists():
                run_dir.mkdir(parents=True, exist_ok=False)
                poster_name = run_name
                output_dir = str(run_dir)
                break
        else:
            raise RuntimeError(f"unable to allocate unique output_dir under {output_root} for base_name={base_name}")
    
    return PosterState(
        pdf_path=pdf_path,
        output_dir=output_dir,
        poster_name=poster_name,
        text_model=_get_model_config(text_model),
        vision_model=_get_model_config(vision_model),
        images=None,
        tables=None,
        narrative=None,
        poster_plan=None,
        poster_width=width,
        poster_height=height,
        wireframe_layout=None,
        content_filled_layout=None,
        final_layout=None,
        narrative_content=None,
        classified_visuals=None,
        structured_sections=None,
        story_board=None,
        curated_content=None,
        design_layout=None,
        section_title_design=None,
        color_scheme=None,
        keywords=None,
        styled_layout=None,
        url=url,
        logo_path=logo_path,
        aff_logo_path=aff_logo_path,
        tokens=TokenUsage(),
        current_agent="init",
        errors=[]
    )


def _get_model_config(model_id: str) -> ModelConfig:
    """get model configuration"""
    # Local OpenAI-compatible proxy for Gemini models (see `gemini_proxy.py`)
    gemini_local_base_url = "http://127.0.0.1:51958/v1"

    configs = {
        "claude": ModelConfig("claude-sonnet-4-20250514", "anthropic"),
        "claude-sonnet-4-20250514": ModelConfig("claude-sonnet-4-20250514", "anthropic"),
        # Route Gemini through local OpenAI-compatible proxy by default.
        "gemini": ModelConfig("gemini-2.5-pro", "openai", base_url=gemini_local_base_url, api_key="EMPTY"),
        "gemini-2.5-pro": ModelConfig("gemini-2.5-pro", "openai", base_url=gemini_local_base_url, api_key="EMPTY"),
        "gpt-4o-2024-08-06": ModelConfig("gpt-4o-2024-08-06", "openai"),
        "gpt-4.1-2025-04-14": ModelConfig("gpt-4.1-2025-04-14", "openai"),
        "gpt-4.1-mini-2025-04-14": ModelConfig("gpt-4.1-mini-2025-04-14", "openai"),
        "glm-4.6": ModelConfig("glm-4.6", "zhipu"),
        "glm-4.5": ModelConfig("glm-4.5", "zhipu"),
        "glm-4.5-air": ModelConfig("glm-4.5-air", "zhipu"),
        "glm-4.5v": ModelConfig("glm-4.5v", "zhipu"),
        "glm-4": ModelConfig("glm-4", "zhipu"),
        "glm-4v": ModelConfig("glm-4v", "zhipu"),
        "kimi-k2-turbo-preview": ModelConfig("kimi-k2-turbo-preview", "moonshot"),
        "moonshot-v1-8k-vision-preview": ModelConfig("moonshot-v1-8k-vision-preview", "moonshot"),
        "MiniMax-M2": ModelConfig("MiniMax-M2", "Minimax"),
        "qwen3-max": ModelConfig("qwen3-max", "Alibaba"),
        "qwen3-vl-plus": ModelConfig("qwen3-vl-plus", "Alibaba"),
    }
    return configs.get(model_id, configs["gpt-4.1-2025-04-14"])