from typing import Optional, List, Tuple, Dict, Any
import logging
import re
from bs4 import BeautifulSoup, NavigableString, Tag

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.langgraph_agents.state import GraphState, ProcessingStep
from src.models.pydantic_models import ValidationResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

class AdvancedHTMLFixerOutput(BaseModel):
    fixed_html: str = Field(..., description="Fixed and valid HTML content")
    fixes_applied: List[str] = Field(default_factory=list, description="List of fixes applied")
    structure_rebuilt: bool = Field(default=False, description="Whether HTML structure was rebuilt")

class AdvancedHTMLFixer:
    def __init__(self):
        self.void_elements = {
            'area', 'base', 'br', 'col', 'embed', 'hr', 'img',
            'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'
        }

    def fix_html_structure(self, html_content: str, target_brand: str = "") -> Tuple[str, List[str]]:
        fixes_applied = []

        try:
            fixed_html = self._preprocess_html(html_content)
            fixes_applied.append("Preprocessed HTML for parsing")

            soup = BeautifulSoup(fixed_html, 'html.parser')

            soup, structure_fixes = self._ensure_basic_structure(soup)
            fixes_applied.extend(structure_fixes)

            soup, tag_fixes = self._fix_tag_issues(soup)
            fixes_applied.extend(tag_fixes)

            soup, attr_fixes = self._fix_attributes(soup)
            fixes_applied.extend(attr_fixes)

            if target_brand:
                soup, brand_fixes = self._replace_brand_names(soup, target_brand)
                fixes_applied.extend(brand_fixes)

            final_html = self._post_process_html(str(soup))
            fixes_applied.append("Applied final HTML cleanup")

            return final_html, fixes_applied

        except Exception as e:
            logger.error(f"HTML fixing failed: {e}")
            return self._create_minimal_valid_html(target_brand), ["Created minimal valid HTML due to parsing errors"]

    def _preprocess_html(self, html_content: str) -> str:
        html = html_content

        non_void_selfclosing = re.findall(r'<(div|span|p|h[1-6]|section|article|header|footer|nav|main|aside|ul|ol|li|table|tr|td|th|tbody|thead|tfoot)\s*([^>]*)/>', html)
        for tag, attrs in non_void_selfclosing:
            html = re.sub(rf'<{tag}\s*{re.escape(attrs)}/>', f'<{tag} {attrs}></{tag}>', html)

        html = re.sub(r'</(meta|img|br|hr|input|link)>', '', html)

        html = re.sub(r'<!DOCTYPE[^>]*>', '', html, count=1)
        if not html.strip().lower().startswith("<!doctype html>"):
            html = '<!DOCTYPE html>\n' + html

        return html

    def _ensure_basic_structure(self, soup: BeautifulSoup) -> Tuple[BeautifulSoup, List[str]]:
        fixes = []

        html_tag = soup.find('html')
        if not html_tag:
            new_html = soup.new_tag('html', lang='ru')
            for element in list(soup.children):
                if element.name != 'html':
                    new_html.append(element.extract())
            soup.clear()
            soup.append(new_html)
            html_tag = new_html
            fixes.append("Created missing html tag")

        head_tag = html_tag.find('head')
        if not head_tag:
            head_tag = soup.new_tag('head')
            html_tag.insert(0, head_tag)
            fixes.append("Created missing head tag")

        if not head_tag.find('meta', charset=True):
            charset_meta = soup.new_tag('meta', charset='UTF-8')
            head_tag.insert(0, charset_meta)
            fixes.append("Added charset meta tag")

        if not head_tag.find('meta', attrs={'name': 'viewport'}):
            viewport_meta = soup.new_tag('meta', attrs={
                'name': 'viewport',
                'content': 'width=device-width, initial-scale=1.0'
            })
            head_tag.append(viewport_meta)
            fixes.append("Added viewport meta tag")

        if not head_tag.find('title'):
            title_tag = soup.new_tag('title')
            title_tag.string = 'Generated Page'
            head_tag.append(title_tag)
            fixes.append("Added title tag")

        body_tag = html_tag.find('body')
        if not body_tag:
            body_tag = soup.new_tag('body')
            for element in list(html_tag.children):
                if element != head_tag and hasattr(element, 'name'):
                    body_tag.append(element.extract())
            html_tag.append(body_tag)
            fixes.append("Created missing body tag")

        return soup, fixes

    def _fix_tag_issues(self, soup: BeautifulSoup) -> Tuple[BeautifulSoup, List[str]]:
        fixes = []

        for p_tag in soup.find_all('p'):
            for block_element in p_tag.find_all(['div', 'section', 'article', 'header', 'footer', 'nav', 'main']):
                p_tag.insert_before(block_element.extract())
                fixes.append(f"Fixed block element {block_element.name} inside p tag")

        for tag in soup.find_all(['header', 'nav', 'main', 'footer', 'section', 'article']):
            if not tag.get_text(strip=True) and not tag.find_all(['img', 'input', 'button', 'form']):
                tag.decompose()
                fixes.append(f"Removed empty {tag.name} tag")

        main_tags = soup.find_all('main')
        if len(main_tags) > 1:
            for main_tag in main_tags[1:]:
                main_tag.unwrap()
            fixes.append("Removed duplicate main tags")

        return soup, fixes

    def _fix_attributes(self, soup: BeautifulSoup) -> Tuple[BeautifulSoup, List[str]]:
        fixes = []

        for img in soup.find_all('img'):
            if not img.get('alt'):
                img['alt'] = 'Image'
                fixes.append("Added alt attribute to img tag")

            if not img.get('src') or img.get('src') == '#':
                img['src'] = 'https://via.placeholder.com/300x200'
                fixes.append("Fixed missing src attribute")

        for link in soup.find_all('a'):
            if not link.get('href'):
                link['href'] = '#'
                fixes.append("Added href attribute to link")

        for element in soup.find_all():
            if element.attrs:
                for attr, value in list(element.attrs.items()):
                    if attr in ['checked', 'disabled', 'readonly', 'selected']:
                        if value not in [True, False, attr]:
                            element.attrs[attr] = attr
                            fixes.append(f"Fixed boolean attribute {attr}")

        return soup, fixes

    def _replace_brand_names(self, soup: BeautifulSoup, target_brand: str) -> Tuple[BeautifulSoup, List[str]]:
        fixes = []
        generic_brands = [
            'Lumina Jewelry', 'Stellar Gems', 'Golden Touch',
            'Diamond Dreams', 'Company Name', 'Brand Name', 'Your Brand'
        ]

        def replace_text_in_element(element):
            if isinstance(element, NavigableString):
                text = str(element)
                for brand in generic_brands:
                    if brand.lower() in text.lower():
                        new_text = re.sub(re.escape(brand), target_brand, text, flags=re.IGNORECASE)
                        element.replace_with(new_text)
                        fixes.append(f"Replaced '{brand}' with '{target_brand}'")
                        return
            elif hasattr(element, 'children'):
                for child in list(element.children):
                    replace_text_in_element(child)

        replace_text_in_element(soup)
        return soup, fixes

    def _post_process_html(self, html: str) -> str:
        html = re.sub(r'\n\s*\n', '\n', html)
        html = re.sub(r'>\s+<', '><', html)
        html = html.replace('&nbsp;', ' ')
        return html

    def _create_minimal_valid_html(self, brand_name: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{brand_name} - Generated Page</title>
</head>
<body>
    <header>
        <h1>{brand_name}</h1>
    </header>
    <main>
        <section>
            <h2>Error: Page Generation Failed</h2>
            <p>A critical error occurred during page generation. Please try again.</p>
        </section>
    </main>
    <footer>
        <p>&copy; 2025 {brand_name}. All rights reserved.</p>
    </footer>
</body>
</html>"""

class HTMLFixingNode:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.html_fixer = AdvancedHTMLFixer()
        self.parser = JsonOutputParser(pydantic_object=AdvancedHTMLFixerOutput)

    async def html_fixing_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing HTML fixing node for {state['brand_name']}")

        if not state.get("final_html"):
            error_msg = "No HTML content to fix."
            logger.error(error_msg)
            return {
                **state,
                "final_html": self.html_fixer._create_minimal_valid_html(state["brand_name"]),
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "html_fixing", "message": error_msg}
                ]
            }

        try:
            fixed_html, fixes_applied = self.html_fixer.fix_html_structure(
                state["final_html"], 
                state["brand_name"]
            )
            
            metrics = state.get("metrics", {})
            metrics["targeted_fixes_applied"] = metrics.get("targeted_fixes_applied", 0) + len(fixes_applied)

            return {
                **state,
                "final_html": fixed_html,
                "metrics": metrics,
                "messages": state.get("messages", []) + [
                    {"type": "info", "step": "html_fixing", "message": f"Applied {len(fixes_applied)} rule-based fixes"}
                ]
            }
            
        except Exception as e:
            error_msg = f"Error in HTML fixing node: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                **state,
                "final_html": self.html_fixer._create_minimal_valid_html(state["brand_name"]),
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "html_fixing", "message": error_msg}
                ]
            }