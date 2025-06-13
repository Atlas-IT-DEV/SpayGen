from typing import List, Dict, Any
import logging

from langchain_openai import ChatOpenAI

from src.langgraph_agents.state import GraphState, ProcessingStep
from src.core.html_chunker import HTMLChunker, ChunkProcessingResult
from src.langgraph_agents.selective_chunk_processor_node import SelectiveChunkProcessor
from src.langgraph_agents.chunk_processor_node import ChunkProcessor
from src.langgraph_agents.html_fixing_node import AdvancedHTMLFixer

from src.utils.logging import get_logger

logger = get_logger(__name__)

class ChunkedModificationNode:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.html_chunker = HTMLChunker()
        self.chunk_processor = ChunkProcessor(llm=self.llm)
        self.selective_chunk_processor = SelectiveChunkProcessor(chunk_processor=self.chunk_processor)
        self.html_fixer = AdvancedHTMLFixer()

    async def chunked_modification_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing chunked template modification node for {state['brand_name']}")

        selected_template = state.get("selected_template")
        generated_content = state.get("generated_content")

        if not selected_template or not generated_content:
            error_msg = "Missing template or generated content for chunked modification."
            logger.error(error_msg)
            fallback_html = selected_template.html if selected_template else ""
            fallback_css = selected_template.css if selected_template else ""
            
            return {
                **state,
                "final_html": fallback_html,
                "final_css": fallback_css,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "template_modification", "message": error_msg}
                ]
            }

        original_html = selected_template.html
        original_css = selected_template.css

        try:
            chunks = self.html_chunker.split_html(original_html, original_css)
            
            chunk_results: List[ChunkProcessingResult] = await self.selective_chunk_processor.process_chunks_with_selective_retry(
                chunks, generated_content, state["spec"], max_retry_attempts=2
            )
            
            metrics = state.get("metrics", {})
            metrics["chunks_processed"] = len(chunk_results)
            metrics["chunks_failed"] = sum(1 for result in chunk_results if result.errors)

            doctype = self.html_fixer._preprocess_html(original_html).split('\n', 1)[0]
            reconstructed_html = self.html_chunker.reconstruct_html(chunk_results, doctype)

            if not reconstructed_html.strip():
                error_msg = "Failed to reconstruct HTML after chunk processing."
                logger.error(error_msg)
                return {
                    **state,
                    "final_html": original_html,
                    "final_css": original_css,
                    "metrics": metrics,
                    "messages": state.get("messages", []) + [
                        {"type": "error", "step": "template_modification", "message": error_msg}
                    ]
                }

            fixed_html, fixes_applied = self.html_fixer.fix_html_structure(reconstructed_html, state["brand_name"])
            metrics["targeted_fixes_applied"] = metrics.get("targeted_fixes_applied", 0) + len(fixes_applied)

            return {
                **state,
                "final_html": fixed_html,
                "final_css": original_css,
                "metrics": metrics,
                "messages": state.get("messages", []) + [
                    {"type": "info", "step": "template_modification", 
                     "message": f"Chunked template modification completed. Processed {metrics['chunks_processed']} chunks, {metrics['chunks_failed']} failed"}
                ]
            }

        except Exception as e:
            error_msg = f"Error in chunked template modification node: {e}"
            logger.error(error_msg, exc_info=True)
            
            return {
                **state,
                "final_html": original_html,
                "final_css": original_css,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "template_modification", "message": error_msg}
                ]
            }