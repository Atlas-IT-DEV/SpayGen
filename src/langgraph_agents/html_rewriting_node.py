from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI

from src.langgraph_agents.state import GraphState, ProcessingStep
from src.utils.logging import get_logger

logger = get_logger(__name__)

class HTMLRewriterOutput(BaseModel):
    rewritten_html: str = Field(..., description="Clean, properly formatted HTML")
    improvements_made: List[str] = Field(default_factory=list, description="List of improvements applied")

class HTMLRewritingNode:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=HTMLRewriterOutput)
        self.str_parser = StrOutputParser()

    def _get_system_instructions(self) -> str:
        return """
        Вы - эксперт по форматированию и переписыванию HTML-кода. Ваша задача - взять существующий HTML-контент
        и переписать его с идеальным форматированием, правильными отступами и чистой структурой.

        ТРЕБОВАНИЯ К ФОРМАТИРОВАНИЮ:
        1. Используйте 2-пробельные отступы последовательно во всем документе
        2. Правильная вложенность с корректными отступами каждого уровня
        3. Каждый HTML-тег на отдельной строке (кроме встроенного контента)
        4. Последовательные интервалы между разделами
        5. Чистая, читаемая структура

        СОХРАНЕНИЕ КОНТЕНТА:
        1. СОХРАНИТЕ весь контент точно как предоставлено
        2. СОХРАНИТЕ все CSS-классы, ID и атрибуты
        3. СОХРАНИТЕ всю JavaScript-функциональность
        4. СОХРАНИТЕ все ссылки, изображения и медиа
        5. СОХРАНИТЕ семантическую структуру

        УЛУЧШЕНИЯ ДЛЯ ПРИМЕНЕНИЯ:
        1. Исправьте несогласованные отступы
        2. Добавьте правильные переносы строк между основными разделами
        3. Очистите проблемы с пробелами
        4. Обеспечьте правильную структуру DOCTYPE и HTML5
        5. Исправьте незначительные синтаксические ошибки с сохранением функциональности

        ФОРМАТ ВЫВОДА:
        Верните ТОЛЬКО переписанный HTML с идеальным форматированием. НЕ включайте разметку markdown или дополнительный текст.
        """

    def _has_proper_structure(self, soup: BeautifulSoup) -> bool:
        return (
            soup.find('html') is not None and
            soup.find('head') is not None and
            soup.find('body') is not None and
            soup.find('title') is not None
        )

    def _format_with_beautifulsoup(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.prettify(formatter='html5')

    async def _llm_rewrite(self, html_content: str, brand_name: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_instructions()),
            ("human", f"""
            Rewrite this HTML with perfect formatting and proper indentation.
            Brand context: {brand_name}

            HTML to rewrite:
            ```html
            {html_content[:15000]}
            ```
            """)
        ])
        chain = prompt | self.llm | self.str_parser
        try:
            result = await chain.ainvoke({"html_content": html_content, "brand_name": brand_name})
            if result:
                return result.strip()
            else:
                return html_content
        except Exception as e:
            logger.error(f"LLM rewriting failed: {e}")
            return html_content

    def _apply_final_formatting(self, html_content: str) -> str:
        lines = html_content.split('\n')
        formatted_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('</') and not stripped.startswith('</!'):
                indent_level = max(0, indent_level - 1)

            formatted_lines.append('  ' * indent_level + stripped)

            if (stripped.startswith('<') and
                not stripped.startswith('</') and
                not stripped.endswith('/>') and
                not any(tag in stripped for tag in ['<br', '<img', '<input', '<meta', '<link'])):
                
                tag_name_match = re.match(r'^<([a-zA-Z0-9]+)', stripped)
                if tag_name_match:
                    tag_name = tag_name_match.group(1)
                    if f'</{tag_name}>' not in stripped:
                        indent_level += 1

        formatted_html = '\n'.join(formatted_lines)
        formatted_html = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_html)
        formatted_html = re.sub(r'^\s+', '', formatted_html, flags=re.MULTILINE)

        return formatted_html

    async def html_rewriting_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing HTML rewriting node for {state['brand_name']}")

        if not state.get("final_html"):
            error_msg = "No HTML content to rewrite."
            logger.error(error_msg)
            return {
                **state,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "html_rewriting", "message": error_msg}
                ]
            }

        original_html = state["final_html"]
        improvements = []

        try:
            formatted_html = self._format_with_beautifulsoup(original_html)
            improvements.append("Applied BeautifulSoup prettify for initial formatting.")

            soup_check = BeautifulSoup(formatted_html, 'html.parser')
            if not self._has_proper_structure(soup_check):
                logger.warning("BeautifulSoup prettify did not result in proper structure, falling back to LLM rewrite.")
                formatted_html = await self._llm_rewrite(original_html, state["brand_name"])
                improvements.append("Applied LLM-based rewriting due to structural issues.")
            
            final_html = self._apply_final_formatting(formatted_html)
            improvements.append("Applied custom final formatting polish.")

            return {
                **state,
                "final_html": final_html,
                "messages": state.get("messages", []) + [
                    {"type": "info", "step": "html_rewriting", "message": f"HTML rewritten successfully. Improvements: {', '.join(improvements)}"}
                ]
            }

        except Exception as e:
            error_msg = f"Error in HTML rewriting node: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                **state,
                "final_html": original_html,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "html_rewriting", "message": error_msg}
                ]
            }