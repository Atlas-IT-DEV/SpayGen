from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from typing_extensions import TypedDict

from src.models.pydantic_models import GeneratedWhitePage, WhitePageSpec, FullPageTemplate, ValidationResult, GeneratedContent

class ProcessingStep(Enum):
    TEMPLATE_SELECTION = "template_selection"
    CONTENT_GENERATION = "content_generation"
    TEMPLATE_MODIFICATION = "template_modification"
    VALIDATION = "validation"
    UNIQUENESS = "uniqueness"
    HTML_REWRITING = "html_rewriting"

class GenerationMetrics(TypedDict, total=False):
    html_size_reduction: float
    validation_score: float
    retry_count: int
    chunks_processed: int
    chunks_failed: int
    targeted_fixes_applied: int
    failed_steps: List[str]
    current_attempt: int

class GraphState(TypedDict, total=False):
    spec: WhitePageSpec
    brand_name: str
    page_generation_id: str
    
    selected_template: Optional[FullPageTemplate]
    generated_content: Optional[GeneratedContent]
    final_html: str
    final_css: str
    final_validation: ValidationResult
    metrics: GenerationMetrics
    original_template_size: int
    
    should_retry_pipeline: bool
    should_apply_targeted_fixes: bool
    should_proceed_to_uniqueness: bool
    
    messages: List[Dict[str, Any]]

def create_initial_state(spec: WhitePageSpec) -> GraphState:
    return GraphState(
        spec=spec,
        brand_name=spec.brand_name,
        page_generation_id=f"page_gen_{spec.brand_name}_{datetime.now().timestamp()}",
        selected_template=None,
        generated_content=None,
        final_html="",
        final_css="",
        final_validation=ValidationResult(
            is_valid=False,
            errors=["Page generation not completed"]
        ),
        metrics=GenerationMetrics(
            html_size_reduction=0.0,
            validation_score=0.0,
            retry_count=0,
            chunks_processed=0,
            chunks_failed=0,
            targeted_fixes_applied=0,
            failed_steps=[],
            current_attempt=0
        ),
        original_template_size=0,
        should_retry_pipeline=False,
        should_apply_targeted_fixes=False,
        should_proceed_to_uniqueness=False,
        messages=[]
    )

def calculate_size_reduction(state: GraphState) -> None:
    if state.get("original_template_size", 0) > 0:
        current_size = len(state.get("final_html", ""))
        reduction = ((state["original_template_size"] - current_size) / state["original_template_size"]) * 100
        if "metrics" not in state:
            state["metrics"] = GenerationMetrics()
        state["metrics"]["html_size_reduction"] = reduction

def get_final_result(state: GraphState) -> GeneratedWhitePage:
    calculate_size_reduction(state)
    return GeneratedWhitePage(
        html=state.get("final_html", ""),
        css=state.get("final_css", ""),
        spec=state["spec"],
        validation=state.get("final_validation", ValidationResult(is_valid=False, errors=["Generation failed"]))
    )