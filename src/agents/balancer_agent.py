"""
column space balancer
"""

import json
from typing import Dict, List, Any
from src.state.poster_state import PosterState
from utils.langgraph_utils import load_prompt_by_column_count, LangGraphAgent, extract_json
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error

class BalancerAgent:
    def __init__(self):
        self.name = "balancer_agent"
        # column_count is read from global config (same as other agents)
        from src.config.poster_config import load_config
        cfg = load_config()
        self.column_count = int(cfg.get("layout", {}).get("column_count", 3))
        if self.column_count <= 0:
            raise ValueError(f"layout.column_count must be a positive integer, got {self.column_count}")
        self.column_ids = list(range(1, self.column_count + 1))
        self.balancer_prompt = load_prompt_by_column_count("layout_balancer.txt", self.column_count)

    def __call__(self, initial_layout_data: Dict, column_analysis: Dict, 
                 state: PosterState) -> Dict:
        """optimize column space distribution"""
        
        log_agent_info(self.name, "optimizing column balance")
        
        structured_sections = state.get("structured_sections")
        story_board = state.get("story_board")
        
        columns = column_analysis.get("columns", [])
        # Support both legacy dict form and new list form.
        if isinstance(columns, dict):
            # Legacy: {"left": {...}, "middle": {...}, ...}
            items = []
            for k, v in columns.items():
                items.append({"column_id": k, **(v or {})})
        else:
            items = list(columns) if isinstance(columns, list) else []

        # Build status markdown for N columns.
        status_lines: List[str] = []
        for col in items:
            col_id = col.get("column_id")
            util = col.get("utilization_rate", 0.0) or 0.0
            status = col.get("status", "unknown")
            status_lines.append(f"- **Column {col_id}**: {util:.1%} utilization - {status}")
        column_status_markdown = "\n".join(status_lines) if status_lines else "- (no column data)"
        log_agent_info(self.name, "utilization summary:\n" + column_status_markdown)
        
        agent = LangGraphAgent("layout optimization specialist", state["text_model"])
        
        variables = {
            "structured_sections": json.dumps(structured_sections, indent=2),
            "current_story_board": json.dumps(story_board, indent=2),
            "column_analysis": json.dumps(column_analysis, indent=2),
            "available_height": column_analysis["available_height"],
            "column_count": self.column_count,
            "column_status_markdown": column_status_markdown,
        }
        
        prompt = self.balancer_prompt.format(**variables)
        response = agent.step(prompt)
        
        log_agent_info(self.name, f"response {len(response.content)} chars")
        
        optimized_story_board = extract_json(response.content)
        
        if not self._validate_story_board(optimized_story_board):
            raise ValueError("optimized story_board validation failed (retry/fallback removed)")
        
        log_agent_success(self.name, "optimized story board created")
        return {
            "optimized_story_board": optimized_story_board,
            "balancer_decisions": self._extract_decisions(response.content),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens
        }

    def _validate_story_board(self, story_board: Dict) -> bool:
        """validate story board structure"""
        if "spatial_content_plan" not in story_board:
            return False
        
        scp = story_board["spatial_content_plan"]
        if "sections" not in scp or not isinstance(scp["sections"], list):
            return False
            
        for section in scp["sections"]:
            if section is None:
                log_agent_error(self.name, "null section found")
                return False
            if not isinstance(section, dict):
                log_agent_error(self.name, f"invalid section type: {type(section)}")
                return False
            if "column_assignment" not in section:
                return False
            col = section.get("column_assignment")
            if not isinstance(col, int) or not (1 <= col <= self.column_count):
                return False
                
        return True

    def _extract_decisions(self, response_content: str) -> Dict:
        """extract optimization decisions from response"""
        decisions = {
            "text_adjustments": [],
            "section_additions": [],
            "section_removals": [],
            "optimizations": []
        }
        
        content_patterns = ["expanded text", "added detail", "enhanced content", "increased content",
                          "reduced text", "shortened", "condensed content", "decreased content"]
        addition_patterns = ["added section", "included section", "new section"]
        removal_patterns = ["removed section", "deleted section", "eliminated section"]
        optimization_patterns = ["within column", "column optimization", "adjusted in", "optimized in"]
        
        for line in response_content.split('\n'):
            line_lower = line.lower()
            if any(p in line_lower for p in content_patterns):
                decisions["text_adjustments"].append(line.strip())
            elif any(p in line_lower for p in addition_patterns):
                decisions["section_additions"].append(line.strip())
            elif any(p in line_lower for p in removal_patterns):
                decisions["section_removals"].append(line.strip())
            elif any(p in line_lower for p in optimization_patterns):
                decisions["optimizations"].append(line.strip())
                
        return decisions


def balancer_agent_node(state: PosterState) -> Dict[str, Any]:
    """balancer agent node for langgraph"""
    agent = BalancerAgent()
    result = agent(state.get("initial_layout_data"), 
                  state.get("column_analysis"), 
                  state)
    
    state["tokens"].add_text(
        result.get("input_tokens", 0),
        result.get("output_tokens", 0)
    )
    
    return {
        **state,
        "optimized_story_board": result["optimized_story_board"],
        "balancer_decisions": result["balancer_decisions"],
        "current_agent": "balancer_agent"
    }