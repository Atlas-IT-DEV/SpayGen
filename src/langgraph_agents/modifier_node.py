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
    SIZE_REDUCTION_THRESHOLD = 0.4

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ModifierAgentCoreOutput)

    def _get_system_instructions(self, is_chunk_mode: bool = False) -> str:
        instructions = """
        Вы - эксперт по модификации HTML/CSS, специализирующийся на замене контента с сохранением структуры.
        Ваша задача - МОДИФИЦИРОВАТЬ содержимое существующего HTML-шаблона.

        СТРАТЕГИЯ ЗАМЕНЫ КОНТЕНТА:
        1. ЗАМЕНЯЙТЕ ТОЛЬКО конкретный placeholder-контент реальными данными из generated_content
        2. Заменяйте placeholder-текст типа "Название компании", "Название продукта" и т.д. реальными значениями
        3. Заменяйте placeholder-изображения реальными URL из generated_content.images
        4. Заменяйте placeholder-контактную информацию реальными данными из generated_content.contact_info
        5. Заменяйте placeholder-списки продуктов реальными элементами из generated_content.items

        ЗАПРЕЩЕННЫЕ ДЕЙСТВИЯ:
        - Не удаляйте элементы навигации или footer
        - Не уменьшайте значительно размер HTML (цель - замена контента, а не удаление)

        ФОРМАТ ВЫВОДА:
        Верните JSON с полями modified_html и modified_css.
        """
        
        if is_chunk_mode:
            instructions += """
        РЕЖИМ ОБРАБОТКИ ФРАГМЕНТОВ:
        При обработке HTML-фрагментов (частичного HTML-контента):
        - Фокусируйтесь только на контенте внутри предоставленного фрагмента
        - Сохраняйте все HTML-теги и атрибуты точно как предоставлено
        - Сохраняйте точные отступы и пробелы из фрагмента
        - Заменяйте только текстовое содержимое и src-атрибуты
        - Возвращайте модифицированный фрагмент без добавления или удаления структурных элементов
        - Сохраняйте тот же стиль форматирования, что и во входном фрагменте
        """
            instructions += "\n\nВерните JSON с полями 'modified_html' и 'modified_css' (CSS может быть пустым, если инлайновый)."
        else:
            instructions += f"\n\nТРЕБУЕМЫЙ ТОЧНЫЙ ФОРМАТ JSON ВЫВОДА:\n{self.parser.get_format_instructions()}"

        return instructions

    def _validate_and_fix_modification_data(
        self,
        data: Dict[str, Any],
        original_html: str,
        original_css: str,
        original_size: int
    ) -> Dict[str, Any]:
        modified_html = data.get("modified_html", "").strip()
        modified_css = data.get("modified_css", "").strip()

        if not modified_html:
            logger.warning("LLM returned empty HTML, using original")
            modified_html = original_html

        if not modified_css:
            logger.warning("LLM returned empty CSS, using original CSS")
            modified_css = original_css

        html_lower = modified_html.lower()
        required_tags = ["<!doctype html>", "<html", "<head>", "<body"]
        missing_tags = [tag for tag in required_tags if tag not in html_lower]

        if missing_tags and not (original_size > 15000):
            logger.warning(f"Modified HTML missing required tags: {missing_tags}, using original")
            modified_html = original_html
            modified_css = original_css

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

    async def template_modification_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing template modification node for {state['brand_name']}")

        selected_template = state.get("selected_template")
        generated_content = state.get("generated_content")

        if not selected_template or not generated_content:
            error_msg = "Missing template or generated content for modification."
            logger.error(error_msg)
            fallback_html = selected_template.html if selected_template else ""
            fallback_css = selected_template.css if selected_template else self.MINIMAL_CSS
            
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
        original_size = len(original_html)

        if not original_css.strip():
            logger.warning("Original CSS is empty, using minimal CSS for modification context.")
            original_css = self.MINIMAL_CSS

        is_chunk_mode = original_size > 15000

        try:
            system_message = self._get_system_instructions(is_chunk_mode=is_chunk_mode)
            
            query_data = {
                "original_html": original_html,
                "original_css": original_css,
                "generated_content": generated_content.model_dump(),
                "spec": state["spec"].model_dump(),
                "preservation_requirements": {
                    "original_size": original_size,
                    "min_size_threshold": int(original_size * self.SIZE_REDUCTION_THRESHOLD),
                    "preserve_structure": True,
                    "preserve_styling": True,
                    "chunk_mode": is_chunk_mode
                }
            }

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", f"Modify the provided HTML and CSS. Original HTML:\n```html\n{original_html}\n```\nOriginal CSS:\n```css\n{original_css}\n```\nGenerated Content:\n```json\n{json.dumps(generated_content.model_dump(), ensure_ascii=False, indent=2)}\n```\nWhitePage Specification:\n```json\n{json.dumps(state['spec'].model_dump(), ensure_ascii=False, indent=2)}\n```\n")
            ])
            
            chain = prompt | self.llm | self.parser
            
            llm_response = await chain.ainvoke(query_data)
            
            validated_data = self._validate_and_fix_modification_data(
                llm_response.model_dump(), original_html, original_css, original_size
            )
            
            return {
                **state,
                "final_html": validated_data["modified_html"],
                "final_css": validated_data["modified_css"],
                "messages": state.get("messages", []) + [
                    {"type": "info", "step": "template_modification", "message": "Template modified successfully"}
                ]
            }

        except Exception as e:
            error_msg = f"Error in template modification node: {e}"
            logger.error(error_msg, exc_info=True)
            
            return {
                **state,
                "final_html": original_html,
                "final_css": original_css,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "template_modification", "message": error_msg}
                ]
            }