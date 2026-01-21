"""
Main workflow pipeline for paper-to-poster generation
"""

import argparse
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# langgraph imports
from langgraph.graph import StateGraph, START, END

from src.state.poster_state import create_state, PosterState, _get_model_config
from src.config.poster_config import load_config
from src.agents.parser import parser_node
from src.agents.curator import curator_node
from src.agents.layout_with_balancer import layout_with_balancer_node as layout_optimizer_node
from src.agents.section_title_designer import section_title_designer_node
from src.agents.color_agent import color_agent_node
from src.agents.font_agent import font_agent_node
from src.agents.renderer import renderer_node
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

def create_workflow_graph() -> StateGraph:
    """create the langgraph workflow"""
    graph = StateGraph(PosterState)
    
    # add all nodes in the workflow
    graph.add_node("parser", parser_node)
    graph.add_node("curator", curator_node)
    graph.add_node("color_agent", color_agent_node)
    graph.add_node("section_title_designer", section_title_designer_node)
    graph.add_node("layout_optimizer", layout_optimizer_node)
    graph.add_node("font_agent", font_agent_node)
    graph.add_node("renderer", renderer_node)
    
    # workflow: parser -> story board -> color -> title design -> layout -> font -> render
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "curator")
    graph.add_edge("curator", "color_agent")
    graph.add_edge("color_agent", "section_title_designer")
    graph.add_edge("section_title_designer", "layout_optimizer")
    graph.add_edge("layout_optimizer", "font_agent")
    graph.add_edge("font_agent", "renderer")
    graph.add_edge("renderer", END)
    
    return graph


def main():
    parser = argparse.ArgumentParser(description="PosterGen: Multi-agent Aesthetic-aware Paper-to-poster generation")
    parser.add_argument("--paper_path", type=str, required=True, help="Path to the PDF paper")
    parser.add_argument("--text_model", type=str, default="gpt-4o-2024-08-06", 
                       choices=["gpt-4o-2024-08-06", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "claude-sonnet-4-20250514", "gemini-2.5-pro", "glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4", "kimi-k2-turbo-preview", "MiniMax-M2", "qwen3-max"],
                       help="Text model for content processing")
    parser.add_argument("--vision_model", type=str, default="gpt-4o-2024-08-06",
                       choices=["gpt-4o-2024-08-06", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "claude-sonnet-4-20250514", "gemini-2.5-pro", "glm-4.5v", "glm-4v", "moonshot-v1-8k-vision-preview", "MiniMax-M2", "qwen3-vl-plus"],
                       help="Vision model for image analysis")
    parser.add_argument("--poster_width", type=float, default=54, help="Poster width in inches")
    parser.add_argument("--poster_height", type=float, default=36, help="Poster height in inches")
    parser.add_argument("--url", type=str, help="URL for QR code on poster") # TODO
    parser.add_argument("--logo", type=str, default="./data/Robustness_Reprogramming_for_Representation_Learning/logo.png", help="Path to conference/journal logo")
    parser.add_argument("--aff_logo", type=str, default="./data/Robustness_Reprogramming_for_Representation_Learning/aff.png", help="Path to affiliation logo")
    
    args = parser.parse_args()
    
    # poster dimensions: fix width to 54", adjust height by ratio
    input_ratio = args.poster_width / args.poster_height
    # check poster ratio: lower bound 1.4 (ISO A paper size), upper bound 2 (human vision limit)
    if input_ratio > 2 or input_ratio < 1.4:
        print(f"❌ Poster ratio is out of range: {input_ratio}. Please use a ratio between 1.4 and 2.")
        return 1
    
    final_width = args.poster_width
    final_height = args.poster_height
    
    # check .env file
    if env_path.exists():
        print(f"✅ .env file found at: {env_path}")
    else:
        print(f"❌ .env file NOT found")
    
    # check api keys (but allow local base_url / explicit api_key in ModelConfig)
    def _is_local_base_url(url: Optional[str]) -> bool:
        if not url:
            return False
        return url.startswith("http://127.0.0.1") or url.startswith("http://localhost")

    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "Minimax": "MINIMAX_API_KEY",
        "Alibaba": "ALIBABA_API_KEY",
    }

    # Use the same model config mapping as runtime (supports local port overrides).
    text_cfg = _get_model_config(args.text_model)
    vision_cfg = _get_model_config(args.vision_model)

    missing = []
    for cfg in (text_cfg, vision_cfg):
        # If model is routed to local proxy, no env key required.
        if _is_local_base_url(cfg.base_url):
            continue
        # If an api_key is explicitly provided in config, no env key required.
        if cfg.api_key:
            continue
        # Otherwise require provider-specific env var (if known).
        env_key = required_keys.get(cfg.provider)
        if env_key and not os.getenv(env_key):
            missing.append(env_key)

    if missing:
        print(f"❌ Missing API keys: {sorted(set(missing))}")
        return 1
    
    # get pdf path
    pdf_path = args.paper_path
    if not pdf_path or not Path(pdf_path).exists():
        print("❌ PDF not found")
        return 1
    
    print(f"🚀 PosterGen Pipeline")
    print(f"📄 PDF: {pdf_path}")
    print(f"🤖 Models: {args.text_model}/{args.vision_model}")
    print(f"📏 Size: {final_width}\" × {final_height:.2f}\"")
    print(f"🏢 Conference Logo: {args.logo}")
    print(f"🏫 Affiliation Logo: {args.aff_logo}")
    
    try:
        # create poster state
        state = create_state(
            pdf_path, args.text_model, args.vision_model, 
            final_width, final_height, 
            args.url, args.logo, args.aff_logo,
        )
        
        log_agent_info("pipeline", "creating workflow graph")
        graph = create_workflow_graph()
        workflow = graph.compile()
        
        log_agent_info("pipeline", "executing workflow")
        final_state = workflow.invoke(state)

        if final_state.get("errors"):
            log_agent_error("pipeline", f"Pipeline errors: {final_state['errors']}")
            return 1
        required_outputs = ["story_board", "design_layout", "color_scheme", "styled_layout"]
        missing = [out for out in required_outputs if not final_state.get(out)]
        if missing:
            log_agent_error("pipeline", f"Missing outputs: {missing}")
            return 1
        
        log_agent_success("pipeline", "Pipeline completed successfully")

        # full pipeline summary
        log_agent_success("pipeline", "Full pipeline complete")
        log_agent_info("pipeline", f"Total tokens: {final_state['tokens'].input_text} → {final_state['tokens'].output_text}")
        
        output_path = Path(final_state["output_dir"]) / f"{final_state['poster_name']}.pptx"
        log_agent_info("pipeline", f"Final poster saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        log_agent_error("pipeline", f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())