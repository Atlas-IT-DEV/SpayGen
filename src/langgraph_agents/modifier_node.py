from typing import Dict, Any, Optional
import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.models.pydantic_models import WhitePageSpec, FullPageTemplate, GeneratedContent
from src.langgraph_agents.state import GraphState, ProcessingStep
from src.utils.logging import get_logger

logger = get_logger(__name__)

def clean_llm_json_output(json_string: str) -> str:
    json_string = json_string.strip()
    if json_string.startswith("```json"):
        json_string = json_string[len("```json"):].strip()
    if json_string.endswith("```"):
        json_string = json_string[:-len("```")].strip()
    return json_string

class ModifierAgentCoreOutput(BaseModel):
    modified_html: str = Field(..., description="The modified HTML content")
    modified_css: str = Field(..., description="The modified CSS content")

class TemplateModificationNode:
    MINIMAL_CSS = "/* Minimal CSS */ body { margin: 0; padding: 0; font-family: Arial, sans-serif; }"
    SIZE_REDUCTION_THRESHOLD = 0.2

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ModifierAgentCoreOutput)

    def _get_system_instructions(self, is_chunk_mode: bool = False) -> str:
        """Generate system instructions for HTML/CSS content modification."""
        
        base_instructions = """
        You are an adaptive HTML/CSS content enhancer with creative freedom to improve templates.
        Transform content intelligently while maintaining the essence of the original design.

        ENHANCEMENT PHILOSOPHY:
        - Replace placeholders with engaging, contextually relevant content
        - Evolve messaging to create compelling user experiences
        - Enhance visual elements for maximum impact and accessibility
        - Integrate brand elements seamlessly into the existing framework
        - Reimagine product showcases to drive engagement

        CREATIVE AUTONOMY:
        - Restructure content flow for optimal user journey
        - Modify styling approaches to better serve the content
        - Experiment with layout improvements within structural constraints
        - Adapt design patterns to current best practices
        - Innovate while respecting brand consistency

        OUTPUT EXPECTATIONS:
        Deliver JSON with modified_html and modified_css fields showcasing your enhancements.
        """
        
        if is_chunk_mode:
            chunk_enhancements = """
            
            FRAGMENT OPTIMIZATION:
            Work within the provided HTML segment with complete creative latitude.
            Enhance content quality while respecting existing architectural patterns.
            Apply improvements that complement the broader page context.
            
            Provide JSON response with 'modified_html' and 'modified_css' fields.
            """
            return base_instructions + chunk_enhancements
        
        return f"{base_instructions}\n\nSTRUCTURED OUTPUT:\n{self.parser.get_format_instructions()}"

    def _validate_and_fix_modification_data(
        self,
        data: Dict[str, Any],
        original_html: str,
        original_css: str,
        original_size: int
    ) -> Dict[str, Any]:
        modified_html = data.get("modified_html", "").strip()
        modified_css = data.get("modified_css", "").strip()

        # if not modified_html:
        #     logger.warning("LLM returned empty HTML, using original")
        #     modified_html = original_html

        # if not modified_css:
        #     logger.warning("LLM returned empty CSS, using original CSS")
        #     modified_css = original_css

        html_lower = modified_html.lower()
        required_tags = ["<!doctype html>", "<html", "<head>", "<body"]
        missing_tags = [tag for tag in required_tags if tag not in html_lower]

        # if missing_tags and not (original_size > 15000):
        #     logger.warning(f"Modified HTML missing required tags: {missing_tags}, using original")
        #     modified_html = original_html
        #     modified_css = original_css

        modified_size = len(modified_html)
        size_ratio = modified_size / original_size if original_size > 0 else 1.0

        if size_ratio < self.SIZE_REDUCTION_THRESHOLD:
            logger.warning(
                f"Excessive size reduction detected ({size_ratio:.2%}), this might indicate an issue."
            )

        return {
            "modified_html": modified_html,
            "modified_css": modified_css
        }

    async def template_modification_node(state: GraphState) -> Dict[str, Any]:
        try:
            original_html = state["original_html"]
            original_css = state.get("original_css", "")
            generated_content = state["generated_content"]
            spec = state["spec"]
            
            if not original_css:
                logger.warning("Original CSS is empty, using minimal CSS for modification context.")
                original_css = "/* Minimal CSS */"
            
            original_size = len(original_html) + len(original_css)
            
            system_message = self._get_system_instructions()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", 
                    "Modify the provided HTML and CSS.\n\n"
                    "Original HTML:\n```html\n{original_html}\n```\n\n"
                    "Original CSS:\n```css\n{original_css}\n```\n\n"
                    "Generated Content:\n```json\n{generated_content_json}\n```\n\n"
                    "WhitePage Specification:\n```json\n{spec_json}\n```\n"
                )
            ])
            
            query_data = {
                "original_html": original_html,
                "original_css": original_css,
                "generated_content_json": json.dumps(generated_content.model_dump(), ensure_ascii=False, indent=2),
                "spec_json": json.dumps(spec.model_dump(), ensure_ascii=False, indent=2)
            }
            
            chain = prompt | self.llm | self.parser
            llm_response = await chain.ainvoke(query_data)
            
            validated_data = self._validate_and_fix_modification_data(
                llm_response.model_dump(), original_html, original_css, original_size
            )
            
            return {
                "modified_html": validated_data["modified_html"],
                "modified_css": validated_data["modified_css"],
                "modification_results": {
                    "success": True,
                    "original_size": original_size,
                    "modified_size": len(validated_data["modified_html"]) + len(validated_data["modified_css"]),
                    "processing_time": time.time() - start_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error in template modification node: {e}")
            return {
                "modified_html": state.get("original_html", ""),
                "modified_css": state.get("original_css", ""),
                "modification_results": {
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time if 'start_time' in locals() else 0
                }
            }