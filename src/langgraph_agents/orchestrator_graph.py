from typing import Dict, Any, Optional, List
import logging
import asyncio

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.models.pydantic_models import WhitePageSpec, GeneratedWhitePage, ValidationResult
from src.tools.qdrant_manager import AsyncQdrantManager
from src.tools.template_loader import TemplateLoader
from src.utils.logging import get_logger

from src.langgraph_agents.state import GraphState, ProcessingStep, create_initial_state, get_final_result
from src.langgraph_agents.template_node import TemplateSelectionNode
from src.langgraph_agents.content_node import ContentGenerationNode
from src.langgraph_agents.modifier_node import TemplateModificationNode
from src.langgraph_agents.chunked_modification_node import ChunkedModificationNode
from src.langgraph_agents.validation_node import ValidationNode, ValidationDecisionMaker
from src.langgraph_agents.targeted_fixing_node import TargetedFixingNode
from src.langgraph_agents.uniqueness_node import UniquenessNode
from src.langgraph_agents.html_rewriting_node import HTMLRewritingNode

from src.models.openai_client_manager import openai_client

logger = get_logger(__name__)

class WhitePageOrchestratorGraph:
    CHUNKING_THRESHOLD = 15000

    def __init__(self):
        self.llm = ChatOpenAI(
            client=openai_client,
            model=settings.model_settings.get("model"),
            temperature=settings.model_settings.get("temperature"),
            max_tokens=settings.model_settings.get("max_tokens")
        )
        
        self.qdrant_manager = AsyncQdrantManager(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout,
            prefer_grpc=settings.qdrant_prefer_grpc,
            max_retries=settings.qdrant_max_retries,
            retry_delay=settings.qdrant_retry_delay,
            collection_name=settings.qdrant_collection_name,
            embedding_dim=settings.embedding_dim
        )
        self.template_loader = TemplateLoader(qdrant_manager=self.qdrant_manager)

        self.template_selection_node_instance = TemplateSelectionNode(
            qdrant_manager=self.qdrant_manager,
            template_loader=self.template_loader
        )
        self.content_generation_node_instance = ContentGenerationNode(llm=self.llm)
        self.template_modification_node_instance = TemplateModificationNode(llm=self.llm)
        self.chunked_modification_node_instance = ChunkedModificationNode(llm=self.llm)
        self.validation_node_instance = ValidationNode(llm=self.llm)
        self.targeted_fixing_node_instance = TargetedFixingNode(llm=self.llm)
        self.uniqueness_node_instance = UniquenessNode(llm=self.llm)
        self.html_rewriting_node_instance = HTMLRewritingNode(llm=self.llm)

        self.compiled_graph = self._build_graph()

    def _route_modification(self, state: GraphState) -> str:
        """Determines whether to use standard or chunked modification."""
        if state.get("original_template_size", 0) > self.CHUNKING_THRESHOLD:
            return "chunked_modification"
        return "standard_modification"

    def _build_graph(self) -> StateGraph:
        """Builds and compiles the Langgraph workflow."""
        workflow = StateGraph(GraphState)

        workflow.add_node("template_selection", self.template_selection_node_instance.template_selection_node)
        workflow.add_node("content_generation", self.content_generation_node_instance.content_generation_node)
        workflow.add_node("standard_modification", self.template_modification_node_instance.template_modification_node)
        workflow.add_node("chunked_modification", self.chunked_modification_node_instance.chunked_modification_node)
        workflow.add_node("validation", self.validation_node_instance.validation_node)
        workflow.add_node("targeted_fixing", self.targeted_fixing_node_instance.targeted_fixing_node)
        workflow.add_node("uniqueness", self.uniqueness_node_instance.uniqueness_node)
        workflow.add_node("html_rewriting", self.html_rewriting_node_instance.html_rewriting_node)

        workflow.add_edge("template_selection", "content_generation")
        workflow.add_conditional_edges(
            "content_generation",
            self._route_modification,
            {
                "standard_modification": "standard_modification",
                "chunked_modification": "chunked_modification"
            }
        )
        workflow.add_edge("standard_modification", "validation")
        workflow.add_edge("chunked_modification", "validation")
        workflow.add_conditional_edges(
            "validation",
            lambda state: "targeted_fixing" if state.get("should_apply_targeted_fixes", False) else "uniqueness"
        )
        workflow.add_edge("targeted_fixing", "validation")
        workflow.add_edge("uniqueness", "html_rewriting")
        workflow.add_edge("html_rewriting", END)

        workflow.set_entry_point("template_selection")

        return workflow.compile()

    async def generate_whitepage(self, spec: WhitePageSpec) -> GeneratedWhitePage:
        """Executes the workflow to generate a WhitePage."""
        initial_state = create_initial_state(spec)
        
        try:
            final_state = await self.compiled_graph.ainvoke(initial_state)
            return get_final_result(final_state)
            
        except Exception as e:
            logger.error(f"Error generating WhitePage: {e}")
            raise

    async def close_connections(self) -> None:
        """Closes all connections (Qdrant, OpenAI, etc.)."""
        await self.qdrant_manager.close()
        await openai_client.close()