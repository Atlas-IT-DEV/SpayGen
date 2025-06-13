from typing import Dict, Any, Optional, List
import logging
import re

from src.models.pydantic_models import WhitePageSpec, FullPageTemplate
from src.tools.qdrant_manager import AsyncQdrantManager
from src.tools.template_loader import TemplateLoader
from src.langgraph_agents.state import GraphState, ProcessingStep
from src.utils.logging import get_logger

logger = get_logger(__name__)

class TemplateSelectionNode:
    def __init__(self, qdrant_manager: AsyncQdrantManager, template_loader: TemplateLoader) -> None:
        self._qdrant = qdrant_manager
        self._template_loader = template_loader
        self.fallback_template = self._create_fallback_template()

    def _create_fallback_template(self) -> FullPageTemplate:
        return FullPageTemplate(
            name="fallback_template",
            html="""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{brand_name}}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        .content {
            background: white;
            padding: 2rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .contact {
            background: #ecf0f1;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        footer {
            background-color: #34495e;
            color: white;
            text-align: center;
            padding: 1rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>{{brand_name}}</h1>
    </header>
    <div class="container">
        <div class="content">
            <h2>Добро пожаловать в {{brand_name}}</h2>
            <p>{{business_description}}</p>
        </div>
        <div class="contact">
            <h3>Контактная информация</h3>
            <p><strong>Email:</strong> {{contact_email}}</p>
            <p><strong>Телефон:</strong> {{contact_phone}}</p>
            <p><strong>Адрес:</strong> {{address}}</p>
        </div>
    </div>
    <footer>
        <p>&copy; 2024 {{brand_name}}. Все права защищены.</p>
    </footer>
</body>
</html>""",
            css="",
            description="Универсальный резервный шаблон для любого типа страницы",
            tags=["fallback", "universal", "basic"]
        )

    async def _select_best_template(self, templates: List[FullPageTemplate], spec: WhitePageSpec) -> FullPageTemplate:
        page_type_scores = {}

        for template in templates:
            score = 0

            if spec.page_type and spec.page_type.value in template.name.lower():
                score += 100

            if spec.page_type and spec.page_type.value in template.description.lower():
                score += 50

            for tag in template.tags:
                if spec.page_type and spec.page_type.value in tag.lower():
                    score += 25

            if spec.page_description:
                desc_words = spec.page_description.lower().split()
                template_text = f"{template.description} {' '.join(template.tags)}".lower()
                matching_words = sum(1 for word in desc_words if word in template_text)
                score += matching_words * 10

            page_type_scores[template.name] = score
            logger.debug(f"Template {template.name} scored: {score}")

        if not page_type_scores:
            logger.warning("No templates scored, returning fallback.")
            return self.fallback_template

        best_template_name = max(page_type_scores, key=page_type_scores.get)
        best_template = next((t for t in templates if t.name == best_template_name), self.fallback_template)

        logger.info(f"Best template selected: {best_template.name} with score: {page_type_scores.get(best_template_name, 0)}")
        return best_template

    def _has_css_styling(self, template: FullPageTemplate) -> bool:
        has_external_css = bool(template.css and template.css.strip())
        has_inline_css = bool(re.search(r'<style[^>]*>.*?</style>', template.html, re.DOTALL | re.IGNORECASE))
        has_style_links = bool(re.search(r'<link[^>]*rel=["\']stylesheet["\'][^>]*>', template.html, re.IGNORECASE))
        return has_external_css or has_inline_css or has_style_links

    def _validate_template_detailed(self, template: FullPageTemplate) -> Dict[str, Any]:
        if not template:
            return {"is_valid": False, "reason": "Template is None"}
        if not template.html:
            return {"is_valid": False, "reason": "Template HTML is empty"}
        if not self._has_css_styling(template):
            return {"is_valid": False, "reason": "Template has no CSS styling (neither external, inline, nor linked)"}

        html_lower = template.html.lower()
        required_tags = ["<!doctype html>", "<html", "<head>", "<body"]
        missing_tags = [tag for tag in required_tags if tag not in html_lower]

        if missing_tags:
            return {"is_valid": False, "reason": f"Missing required HTML tags: {missing_tags}"}

        logger.debug(f"Template {template.name} validation passed")
        return {"is_valid": True, "reason": "Template is valid"}

    async def template_selection_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing template selection node for {state['brand_name']}")
        
        try:
            spec = state["spec"]
            search_results = await self._template_loader.search_full_page_templates(spec, limit=5)

            if not search_results:
                logger.warning("No templates found in search, using fallback")
                selected_template = self.fallback_template
                message = "No templates found, using fallback."
                message_type = "warning"
            else:
                best_template = await self._select_best_template(search_results, spec)
                validation_result = self._validate_template_detailed(best_template)
                
                if not validation_result["is_valid"]:
                    logger.warning(f"Selected template {best_template.name} failed validation: {validation_result['reason']}, using fallback.")
                    selected_template = self.fallback_template
                    message = f"Selected template failed validation: {validation_result['reason']}, using fallback."
                    message_type = "warning"
                else:
                    selected_template = best_template
                    message = f"Selected template: {best_template.name}"
                    message_type = "info"
            
            original_template_size = len(selected_template.html)

            return {
                **state,
                "selected_template": selected_template,
                "original_template_size": original_template_size,
                "messages": state.get("messages", []) + [
                    {"type": message_type, "step": "template_selection", "message": message}
                ]
            }

        except Exception as e:
            error_msg = f"Error in template selection node: {e}"
            logger.error(error_msg, exc_info=True)
            
            return {
                **state,
                "selected_template": self.fallback_template,
                "original_template_size": len(self.fallback_template.html),
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "template_selection", "message": error_msg}
                ]
            }