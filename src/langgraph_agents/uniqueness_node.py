from typing import Dict, Any, Optional
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.models.pydantic_models import WhitePageSpec
from src.langgraph_agents.state import GraphState
from src.utils.logging import get_logger

logger = get_logger(__name__)


class UniquenessOutput(BaseModel):
    html_content: str = Field(..., description="Unique HTML content")
    css_content: str = Field(..., description="Unique CSS content")


class HTMLSanitizer:
    
    @staticmethod
    def extract_safe_content(html_content: str) -> Dict[str, str]:
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            style_blocks = []
            for style_tag in soup.find_all('style'):
                if style_tag.string:
                    style_blocks.append(style_tag.string)
                style_tag.decompose()
            
            script_blocks = []
            for script_tag in soup.find_all('script'):
                if script_tag.string and 'application/ld+json' not in script_tag.get('type', ''):
                    script_blocks.append(script_tag.string)
                script_tag.decompose()
            
            clean_html = str(soup)
            
            return {
                'clean_html': clean_html,
                'style_content': ' '.join(style_blocks),
                'script_content': ' '.join(script_blocks),
                'original_length': len(html_content)
            }
            
        except Exception as e:
            logger.warning(f"Error sanitizing HTML: {e}")
            return {
                'clean_html': html_content[:5000],
                'style_content': '',
                'script_content': '',
                'original_length': len(html_content)
            }

    @staticmethod
    def reconstruct_html(clean_html: str, style_content: str, script_content: str) -> str:
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(clean_html, 'html.parser')
            
            if style_content and soup.head:
                style_tag = soup.new_tag('style')
                style_tag.string = style_content
                soup.head.append(style_tag)
            
            if script_content and soup.body:
                script_tag = soup.new_tag('script')
                script_tag.string = script_content
                soup.body.append(script_tag)
            
            return str(soup)
            
        except Exception as e:
            logger.warning(f"Error reconstructing HTML: {e}")
            return clean_html


class UniquenessTransformer:
    
    @staticmethod
    def apply_basic_transformations(html_content: str) -> str:
        transformations = [
            (r'class="([^"]*)"', lambda m: f'class="{m.group(1).replace(" ", "  ")}"'),
            (r'<div>', '<div >'),
            (r'<span>', '<span >'),
            (r'>\s+<', '><'),
            (r'\s{2,}', ' '),
        ]
        
        result = html_content
        for pattern, replacement in transformations:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)
        
        return result

    @staticmethod
    def apply_css_transformations(css_content: str) -> str:
        if not css_content or css_content.strip() == "/* No CSS content to process */":
            return css_content
        
        transformations = [
            (r';\s*', '; '),
            (r'{\s*', '{ '),
            (r'\s*}', ' }'),
            (r',\s*', ', '),
        ]
        
        result = css_content
        for pattern, replacement in transformations:
            result = re.sub(pattern, replacement, result)
        
        return result


class UniquenessNode:
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=UniquenessOutput)
        self.sanitizer = HTMLSanitizer()
        self.transformer = UniquenessTransformer()

    async def uniqueness_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing uniqueness node for {state['brand_name']}")

        final_html = state.get("final_html", "")
        final_css = state.get("final_css", "")

        if not final_html:
            error_msg = "Missing HTML content for uniqueness processing."
            logger.error(error_msg)
            return self._create_error_response(state, error_msg)

        if not final_css:
            logger.info("No CSS content found, proceeding with HTML-only uniqueness")
            final_css = "/* No CSS content to process */"

        try:
            if len(final_html) < 10000:
                return await self._apply_llm_uniqueness(state, final_html, final_css)
            else:
                return self._apply_rule_based_uniqueness(state, final_html, final_css)

        except Exception as e:
            error_msg = f"Error in uniqueness node: {e}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(state, error_msg)

    async def _apply_llm_uniqueness(
        self, 
        state: GraphState, 
        html_content: str, 
        css_content: str
    ) -> Dict[str, Any]:
        
        safe_content = self.sanitizer.extract_safe_content(html_content)
        
        system_message = """
        Вы - агент уникализации HTML/CSS. Преобразуйте предоставленный контент, чтобы сделать его уникальным, сохраняя функциональность.

        ТЕХНИКИ ПРЕОБРАЗОВАНИЯ:
        - Переупорядочивание HTML-атрибутов
        - Добавление вариаций пробелов
        - Переименование CSS-классов и ID
        - Изменение порядка CSS-свойств
        - Добавление безвредных CSS-комментариев
        - Модификация паттернов отступов

        ТРЕБОВАНИЯ:
        - Сохранить всю функциональность
        - Поддерживать визуальное оформление
        - Оставить информацию о бренде нетронутой
        - Вернуть валидный HTML и CSS

        ВЫВОД: JSON с полями html_content и css_content.
        """

        spec_data = state["spec"]
        content_summary = f"""
        Преобразуйте этот контент для уникальности:

        Бренд: {spec_data.brand_name}
        Бизнес: {spec_data.business_description}

        Сводка HTML-структуры:
        - Длина: {safe_content['original_length']} символов
        - Есть встроенные стили: {'Да' if safe_content['style_content'] else 'Нет'}
        - Есть скрипты: {'Да' if safe_content['script_content'] else 'Нет'}

        Очищенный HTML-контент:
        {safe_content['clean_html'][:3000]}

        CSS-контент:
        {css_content}

        Примените преобразования уникальности, сохраняя бренд "{spec_data.brand_name}" и контактную информацию.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", content_summary)
        ])
        
        chain = prompt | self.llm | self.parser
        uniqueness_output = await chain.ainvoke({})
        
        unique_html = uniqueness_output.html_content
        unique_css = uniqueness_output.css_content

        if not unique_html.strip() or len(unique_html) < len(html_content) * 0.5:
            logger.warning("LLM output seems incomplete, applying rule-based fallback")
            return self._apply_rule_based_uniqueness(state, html_content, css_content)

        if safe_content['style_content'] and unique_html:
            unique_html = self.sanitizer.reconstruct_html(
                unique_html, 
                safe_content['style_content'], 
                safe_content['script_content']
            )

        return self._create_success_response(state, unique_html, unique_css)

    def _apply_rule_based_uniqueness(
        self, 
        state: GraphState, 
        html_content: str, 
        css_content: str
    ) -> Dict[str, Any]:
        
        logger.info("Applying rule-based uniqueness transformations")
        
        unique_html = self.transformer.apply_basic_transformations(html_content)
        unique_css = self.transformer.apply_css_transformations(css_content)
        
        return self._create_success_response(state, unique_html, unique_css)

    def _create_success_response(
        self, 
        state: GraphState, 
        html_content: str, 
        css_content: str
    ) -> Dict[str, Any]:
        
        return {
            **state,
            "final_html": html_content,
            "final_css": css_content,
            "messages": state.get("messages", []) + [
                {"type": "info", "step": "uniqueness", "message": "Uniqueness applied successfully"}
            ]
        }

    def _create_error_response(self, state: GraphState, error_msg: str) -> Dict[str, Any]:
        return {
            **state,
            "messages": state.get("messages", []) + [
                {"type": "error", "step": "uniqueness", "message": error_msg}
            ]
        }