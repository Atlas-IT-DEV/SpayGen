from typing import Dict, Any, Optional, List, Union
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.models.pydantic_models import WhitePageSpec, ValidationResult
from src.langgraph_agents.state import GraphState
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationDecisionMaker:
    CRITICAL_THRESHOLD = 0.5
    ACCEPTABLE_THRESHOLD = 0.7

    @classmethod
    def should_apply_targeted_fixes(cls, validation_result: ValidationResult) -> bool:
        score = validation_result.score
        error_count = len(validation_result.errors)

        if score >= cls.ACCEPTABLE_THRESHOLD:
            return False

        if score < cls.CRITICAL_THRESHOLD and error_count > 8:
            return False

        critical_errors = [
            error for error in validation_result.errors
            if any(keyword in error.lower() for keyword in ['corrupt', 'malformed', 'invalid syntax', 'broken'])
        ]

        minor_errors = [
            error for error in validation_result.errors
            if any(keyword in error.lower() for keyword in ['contact', 'brand', 'email', 'phone', 'title', 'alt'])
        ]

        return len(critical_errors) <= 2 and (len(minor_errors) >= len(validation_result.errors) * 0.6)


class HTMLAnalyzer:
    @staticmethod
    def extract_content_info(html_content: str) -> Dict[str, Any]:
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for tag in soup(['style', 'script']):
                tag.decompose()
            
            analysis = {
                'title': soup.find('title').get_text() if soup.find('title') else "No title",
                'meta_description': '',
                'headings': [],
                'links_info': [],
                'images_info': [],
                'text_content': soup.get_text()[:2000],
                'has_doctype': html_content.strip().lower().startswith('<!doctype'),
                'has_basic_structure': bool(soup.find('html') and soup.find('head') and soup.find('body'))
            }
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                analysis['meta_description'] = meta_desc.get('content', '')
            
            for i in range(1, 7):
                headings = soup.find_all(f'h{i}')
                for h in headings[:3]:
                    analysis['headings'].append(h.get_text().strip())
            
            links = soup.find_all('a', href=True)
            for link in links[:5]:
                link_text = link.get_text().strip()
                link_href = link['href']
                analysis['links_info'].append(f"Text: {link_text}, URL: {link_href}")
            
            images = soup.find_all('img')
            for img in images[:3]:
                img_src = img.get('src', '')
                img_alt = img.get('alt', '')
                analysis['images_info'].append(f"Source: {img_src}, Alt: {img_alt}")
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error extracting HTML content: {e}")
            return {
                'title': "Error extracting title",
                'meta_description': '',
                'headings': [],
                'links_info': [],
                'images_info': [],
                'text_content': html_content[:1000] if html_content else '',
                'has_doctype': False,
                'has_basic_structure': False
            }

    @staticmethod
    def format_list_for_prompt(items: List[str]) -> str:
        if not items:
            return "None"
        return " | ".join(items[:5])


class ValidationResultParser:
    @staticmethod
    def create_validation_result(response: Union[Dict[str, Any], ValidationResult]) -> ValidationResult:
        if isinstance(response, ValidationResult):
            return response
        
        if not isinstance(response, dict):
            logger.warning(f"Unexpected validation response type: {type(response)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Unexpected response type: {type(response)}"],
                warnings=[],
                score=0.0
            )
        
        try:
            if 'ValidationResult' in response:
                validation_data = response['ValidationResult']
            else:
                validation_data = response
            
            normalized_data = {
                'is_valid': bool(validation_data.get('is_valid', False)),
                'errors': validation_data.get('errors', []),
                'warnings': validation_data.get('warnings', []),
                'score': float(validation_data.get('score', 0.0))
            }
            
            if not isinstance(normalized_data['errors'], list):
                normalized_data['errors'] = [str(normalized_data['errors'])]
            
            if not isinstance(normalized_data['warnings'], list):
                normalized_data['warnings'] = [str(normalized_data['warnings'])]
            
            return ValidationResult(**normalized_data)
            
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            logger.warning(f"Failed to create ValidationResult from response: {e}")
            logger.debug(f"Response structure: {response}")
            
            fallback_score = 0.0
            try:
                if isinstance(response, dict):
                    if 'ValidationResult' in response and 'score' in response['ValidationResult']:
                        fallback_score = float(response['ValidationResult']['score'])
                    elif 'score' in response:
                        fallback_score = float(response['score'])
            except (ValueError, TypeError):
                pass
            
            return ValidationResult(
                is_valid=fallback_score >= 0.7,
                errors=[f"Failed to parse validation response: {str(e)}"],
                warnings=["Response parsing issue - using fallback validation"],
                score=fallback_score
            )


class ValidationNode:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ValidationResult)
        self.html_analyzer = HTMLAnalyzer()
        self.result_parser = ValidationResultParser()

    async def validation_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing validation node for {state['brand_name']}")

        final_html = state.get("final_html")
        if not final_html:
            error_msg = "No HTML content to validate."
            logger.error(error_msg)
            validation_result = ValidationResult(is_valid=False, errors=[error_msg], score=0.0)
            
            return {
                **state,
                "final_validation": validation_result,
                "should_retry_pipeline": True,
                "should_apply_targeted_fixes": False,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "validation", "message": error_msg}
                ]
            }

        try:
            html_analysis = self.html_analyzer.extract_content_info(final_html)
            spec_data = state["spec"]
            
            system_message = """
Вы - эксперт по валидации HTML. Проанализируйте предоставленную HTML-структуру и информацию о контенте 
в соответствии со спецификацией WhitePage для определения валидности, качества и соответствия.

КРИТЕРИИ ВАЛИДАЦИИ:
1. HTML-структура: Полный HTML5-документ с doctype, html, head, body
2. Релевантность контента: Контент соответствует бренду, описанию бизнеса, продуктам
3. Уникальность: Отсутствие общих заполнителей или повторяющихся фраз
4. Контактная информация: Валидная и уникальная контактная информация
5. Качество контента: Подходящий для типа бизнеса и аудитории
6. Техническое соответствие: Правильная семантическая структура

ФОРМАТ ВЫВОДА:
Верните JSON с этими точными полями:
- is_valid: булевое значение (true если score >= 0.7)
- errors: массив конкретных описаний ошибок
- warnings: массив описаний предупреждений
- score: число с плавающей точкой от 0.0 до 1.0 (0.7+ считается приемлемым)

Не оборачивайте JSON в дополнительную структуру или markdown.
"""

            products_text = " | ".join(spec_data.products) if spec_data.products else "None specified"
            headings_text = self.html_analyzer.format_list_for_prompt(html_analysis['headings'])
            links_text = self.html_analyzer.format_list_for_prompt(html_analysis['links_info'])
            images_text = self.html_analyzer.format_list_for_prompt(html_analysis['images_info'])

            validation_data = f"""
СПЕЦИФИКАЦИЯ ДЛЯ ВАЛИДАЦИИ:
Название бренда: {spec_data.brand_name}
Бизнес: {spec_data.business_description}
Ожидаемый Email: {spec_data.contact_email}
Ожидаемый телефон: {spec_data.contact_phone}
Ожидаемый адрес: {spec_data.address}
Тип страницы: {spec_data.page_type.value if spec_data.page_type else 'общий'}
Ожидаемые продукты: {products_text}

РЕЗУЛЬТАТЫ АНАЛИЗА HTML:
Заголовок документа: {html_analysis['title']}
Мета-описание: {html_analysis['meta_description']}
Есть DOCTYPE: {html_analysis['has_doctype']}
Есть базовая структура: {html_analysis['has_basic_structure']}
Основные заголовки: {headings_text}
Найденные ссылки: {links_text}
Информация об изображениях: {images_text}

ОБРАЗЕЦ ТЕКСТОВОГО КОНТЕНТА:
{html_analysis['text_content']}

ЗАДАЧИ ВАЛИДАЦИИ:
1. Проверить, что название бренда "{spec_data.brand_name}" присутствует в заголовке и контенте
2. Проверить соответствие контактной информации: {spec_data.contact_email}, {spec_data.contact_phone}
3. Подтвердить правильную HTML-структуру (DOCTYPE, html, head, body)
4. Оценить уникальность контента и соответствие бизнесу
5. Проверить на запрещенные категории контента
6. Убедиться, что у изображений есть alt-атрибуты
7. Обеспечить использование семантической разметки

Верните результаты валидации в JSON-формате с указанными полями.
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", validation_data)
            ])
            
            chain = prompt | self.llm | self.parser
            response = await chain.ainvoke({})
            
            validation_result = self.result_parser.create_validation_result(response)
            
            score = validation_result.score
            metrics = state.get("metrics", {})
            metrics["validation_score"] = score

            should_apply_fixes = False
            should_retry = False
            should_proceed = False

            if not validation_result.is_valid:
                if ValidationDecisionMaker.should_apply_targeted_fixes(validation_result):
                    should_apply_fixes = True
                else:
                    should_retry = True
            else:
                should_proceed = True

            return {
                **state,
                "final_validation": validation_result,
                "metrics": metrics,
                "should_apply_targeted_fixes": should_apply_fixes,
                "should_retry_pipeline": should_retry,
                "should_proceed_to_uniqueness": should_proceed,
                "messages": state.get("messages", []) + [
                    {"type": "info", "step": "validation", 
                     "message": f"Validation completed. Valid: {validation_result.is_valid}, Score: {score:.2f}"}
                ]
            }

        except Exception as e:
            error_msg = f"Error in validation node: {e}"
            logger.error(error_msg, exc_info=True)
            validation_result = ValidationResult(is_valid=False, errors=[error_msg], score=0.0)
            
            return {
                **state,
                "final_validation": validation_result,
                "should_retry_pipeline": True,
                "should_apply_targeted_fixes": False,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "validation", "message": error_msg}
                ]
            }