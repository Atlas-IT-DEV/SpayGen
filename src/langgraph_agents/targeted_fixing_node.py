from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import re
from abc import ABC, abstractmethod

from bs4 import BeautifulSoup, Tag
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.models.pydantic_models import WhitePageSpec, ValidationResult
from src.langgraph_agents.state import GraphState
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SeverityLevel(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class IssueType(Enum):
    BRAND_MISMATCH = "brand_mismatch"
    CONTACT_MISMATCH = "contact_mismatch"
    ADDRESS_MISMATCH = "address_mismatch"
    PRODUCT_MISMATCH = "product_mismatch"
    LANGUAGE_MISMATCH = "language_mismatch"


@dataclass(frozen=True)
class ValidationIssue:
    issue_type: IssueType
    severity: SeverityLevel
    description: str
    suggested_fix: str
    affected_selectors: Tuple[str, ...]
    search_patterns: Tuple[str, ...]
    replacement_values: Dict[str, str]


@dataclass(frozen=True)
class ValidationContext:
    original_errors: Tuple[str, ...]
    validation_score: float
    brand_name: str
    target_fixes: Dict[str, str]
    retry_attempt: int


class HTMLProcessor:
    """Utility class for HTML processing operations."""
    
    @staticmethod
    def safe_select(soup: BeautifulSoup, selector: str) -> List[Tag]:
        try:
            return soup.select(selector)
        except Exception as e:
            logger.debug(f"Selector {selector} failed: {e}")
            return []
    
    @staticmethod
    def update_element_text(element: Tag, new_text: str) -> None:
        if element.string:
            element.string = new_text
        elif element.get_text():
            element.clear()
            element.string = new_text
    
    @staticmethod
    def find_elements_by_pattern(soup: BeautifulSoup, pattern: str, tags: List[str] = None) -> List[Tag]:
        tags = tags or ['a', 'span', 'div', 'p']
        return soup.find_all(tags, string=re.compile(pattern))


class ValidationAnalyzer:
    """Analyzes validation errors and creates targeted fix strategies."""
    
    _GENERIC_BRANDS = frozenset([
        "Lumina Jewelry", "Stellar Gems", "Golden Touch", 
        "Diamond Dreams", "Company Name"
    ])
    
    _BRAND_SELECTORS = (
        "h1", "h2", "h3", ".brand-name", ".company-name", 
        "title", ".logo-text"
    )
    
    _CONTACT_SELECTORS = (
        ".contact", ".email", ".phone", ".contact-info", 
        "a[href^='mailto:']", "a[href^='tel:']"
    )
    
    _PRODUCT_SELECTORS = (
        ".products", ".items", ".services", ".catalog", ".product-list"
    )
    
    def __init__(self, spec: WhitePageSpec):
        self.spec = spec
        self._error_handlers = {
            "brand": self._create_brand_issue,
            "contact": self._create_contact_issue,
            "email": self._create_contact_issue,
            "phone": self._create_contact_issue,
            "product": self._create_product_issue,
            "address": self._create_address_issue,
            "language": self._create_language_issue,
            "russian": self._create_language_issue,
            "english": self._create_language_issue,
        }

    def analyze_validation_errors(
        self, 
        validation_result: ValidationResult, 
        html_content: str
    ) -> List[ValidationIssue]:
        issues = []
        for error in validation_result.errors:
            issue = self._categorize_and_create_issue(error, html_content)
            if issue:
                issues.append(issue)
        return issues

    def _categorize_and_create_issue(
        self, 
        error: str, 
        html_content: str
    ) -> Optional[ValidationIssue]:
        error_lower = error.lower()
        
        for keyword, handler in self._error_handlers.items():
            if keyword in error_lower:
                return handler(error, html_content)
        
        return None

    def _create_brand_issue(self, error: str, html_content: str) -> ValidationIssue:
        found_brands = tuple(
            brand for brand in self._GENERIC_BRANDS 
            if brand.lower() in html_content.lower()
        )
        
        return ValidationIssue(
            issue_type=IssueType.BRAND_MISMATCH,
            severity=SeverityLevel.CRITICAL,
            description=error,
            suggested_fix=f"Replace generic brand names with '{self.spec.brand_name}'",
            affected_selectors=self._BRAND_SELECTORS,
            search_patterns=found_brands,
            replacement_values={brand: self.spec.brand_name for brand in found_brands}
        )

    def _create_contact_issue(self, error: str, html_content: str) -> ValidationIssue:
        return ValidationIssue(
            issue_type=IssueType.CONTACT_MISMATCH,
            severity=SeverityLevel.MAJOR,
            description=error,
            suggested_fix="Update contact information to match specification",
            affected_selectors=self._CONTACT_SELECTORS,
            search_patterns=(r"[\w\.-]+@[\w\.-]+\.\w+", r"\+?[\d\s\-\(\)]+"),
            replacement_values={
                "email": self.spec.contact_email, 
                "phone": self.spec.contact_phone
            }
        )

    def _create_address_issue(self, error: str, html_content: str) -> ValidationIssue:
        return ValidationIssue(
            issue_type=IssueType.ADDRESS_MISMATCH,
            severity=SeverityLevel.MAJOR,
            description=error,
            suggested_fix="Update address to match specification",
            affected_selectors=(".address", ".location", ".contact-address"),
            search_patterns=("address", "location"),
            replacement_values={"address": self.spec.address}
        )

    def _create_product_issue(self, error: str, html_content: str) -> ValidationIssue:
        products_text = ", ".join(self.spec.products[:3]) if self.spec.products else ""
        
        return ValidationIssue(
            issue_type=IssueType.PRODUCT_MISMATCH,
            severity=SeverityLevel.MINOR,
            description=error,
            suggested_fix="Update product listings to include specified products",
            affected_selectors=self._PRODUCT_SELECTORS,
            search_patterns=("generic product", "sample item"),
            replacement_values={"products": products_text}
        )

    def _create_language_issue(self, error: str, html_content: str) -> ValidationIssue:
        return ValidationIssue(
            issue_type=IssueType.LANGUAGE_MISMATCH,
            severity=SeverityLevel.MAJOR,
            description=error,
            suggested_fix="Translate content to Russian",
            affected_selectors=("p", "span", ".description", ".content", "li"),
            search_patterns=("English text patterns",),
            replacement_values={}
        )


class BaseFixer(ABC):
    """Abstract base class for HTML fixers."""
    
    @abstractmethod
    async def fix(
        self, 
        html_content: str, 
        issue: ValidationIssue, 
        spec: WhitePageSpec
    ) -> str:
        pass


class BrandFixer(BaseFixer):
    """Handles brand name related fixes."""
    
    async def fix(
        self, 
        html_content: str, 
        issue: ValidationIssue, 
        spec: WhitePageSpec
    ) -> str:
        fixed_html = html_content
        
        for old_brand in issue.search_patterns:
            if old_brand in fixed_html:
                fixed_html = re.sub(
                    re.escape(old_brand), 
                    spec.brand_name, 
                    fixed_html, 
                    flags=re.IGNORECASE
                )
        
        soup = BeautifulSoup(fixed_html, 'html.parser')
        
        for selector in issue.affected_selectors:
            elements = HTMLProcessor.safe_select(soup, selector)
            for element in elements:
                if element.string and any(
                    pattern.lower() in element.string.lower() 
                    for pattern in issue.search_patterns
                ):
                    HTMLProcessor.update_element_text(element, spec.brand_name)
        
        return str(soup)


class ContactFixer(BaseFixer):
    """Handles contact information related fixes."""
    
    _EMAIL_PATTERN = r"[\w\.-]+@[\w\.-]+\.\w+"
    _PHONE_PATTERN = r"\+?[\d\s\-\(\)]{10,}"
    
    async def fix(
        self, 
        html_content: str, 
        issue: ValidationIssue, 
        spec: WhitePageSpec
    ) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        self._fix_email_elements(soup, spec.contact_email)
        self._fix_phone_elements(soup, spec.contact_phone)
        
        return str(soup)
    
    def _fix_email_elements(self, soup: BeautifulSoup, email: str) -> None:
        email_elements = HTMLProcessor.find_elements_by_pattern(
            soup, self._EMAIL_PATTERN
        )
        
        for element in email_elements:
            if element.string and '@' in element.string and 'example.com' not in element.string:
                HTMLProcessor.update_element_text(element, email)
                if element.name == 'a':
                    element['href'] = f"mailto:{email}"
    
    def _fix_phone_elements(self, soup: BeautifulSoup, phone: str) -> None:
        phone_elements = HTMLProcessor.find_elements_by_pattern(
            soup, self._PHONE_PATTERN
        )
        
        for element in phone_elements:
            if element.string and any(char.isdigit() for char in element.string):
                HTMLProcessor.update_element_text(element, phone)
                if element.name == 'a':
                    element['href'] = f"tel:{phone}"


class AddressFixer(BaseFixer):
    """Handles address related fixes."""
    
    async def fix(
        self, 
        html_content: str, 
        issue: ValidationIssue, 
        spec: WhitePageSpec
    ) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for selector in issue.affected_selectors:
            elements = HTMLProcessor.safe_select(soup, selector)
            for element in elements:
                if element.get_text():
                    HTMLProcessor.update_element_text(element, spec.address)
        
        return str(soup)


class ProductFixer(BaseFixer):
    """Handles product listing related fixes."""
    
    async def fix(
        self, 
        html_content: str, 
        issue: ValidationIssue, 
        spec: WhitePageSpec
    ) -> str:
        if not spec.products:
            return html_content
        
        soup = BeautifulSoup(html_content, 'html.parser')
        product_sections = soup.select(', '.join(issue.affected_selectors))
        
        for section in product_sections:
            self._update_product_list(section, spec.products)
        
        return str(soup)
    
    def _update_product_list(self, section: Tag, products: List[str]) -> None:
        product_list = section.find('ul') or section.find('ol')
        if not product_list:
            return
        
        existing_items = product_list.find_all('li')
        if len(existing_items) >= 3:
            for i, product in enumerate(products[:len(existing_items)]):
                if i < len(existing_items):
                    HTMLProcessor.update_element_text(existing_items[i], product)


class LLMFixer(BaseFixer):
    """Uses LLM for complex fixes like language translation."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.str_parser = StrOutputParser()
    
    async def fix(
        self, 
        html_content: str, 
        issue: ValidationIssue, 
        spec: WhitePageSpec
    ) -> str:
        if issue.issue_type == IssueType.LANGUAGE_MISMATCH:
            return await self._fix_language_issues(html_content, spec)
        
        return html_content
    
    async def _fix_language_issues(self, html_content: str, spec: WhitePageSpec) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Исправьте языковые проблемы в данном HTML, переведя контент на язык, соответствующий языку бренда и описания бизнеса.
            Сосредоточьтесь на сохранении HTML-структуры при обновлении текстового содержимого.
            Используйте название бренда: {brand_name}
            Описание бизнеса: {business_description}

            Определите язык из названия бренда и описания бизнеса, затем переведите весь текстовый контент на этот язык.
            Сохраняйте неизменными:
            - HTML-теги и атрибуты
            - CSS-классы и ID
            - JavaScript-код
            - URL и email-адреса
            - Технические термины и названия

            Верните ТОЛЬКО исправленный HTML с контентом на соответствующем языке. НЕ включайте разметку markdown или дополнительный текст.
            """),
            ("human", """
            HTML-контент (первые 8000 символов):
            ```html
            {html_content}
            ```
            """)
        ])
        
        chain = prompt | self.llm | self.str_parser
        
        try:
            result = await chain.ainvoke({
                "html_content": html_content[:8000],
                "brand_name": spec.brand_name,
                "business_description": spec.business_description
            })
            
            if result and len(result.strip()) > 1000:
                return result.strip()
        except Exception as e:
            logger.error(f"LLM language fix failed: {e}")
        
        return html_content


class FixerFactory:
    """Factory for creating appropriate fixer instances."""
    
    def __init__(self, llm: ChatOpenAI):
        self._fixers = {
            IssueType.BRAND_MISMATCH: BrandFixer(),
            IssueType.CONTACT_MISMATCH: ContactFixer(),
            IssueType.ADDRESS_MISMATCH: AddressFixer(),
            IssueType.PRODUCT_MISMATCH: ProductFixer(),
            IssueType.LANGUAGE_MISMATCH: LLMFixer(llm),
        }
    
    def get_fixer(self, issue_type: IssueType) -> BaseFixer:
        return self._fixers.get(issue_type, BrandFixer())


class TargetedHTMLFixer:
    """Main class for applying targeted HTML fixes."""
    
    def __init__(self, llm: ChatOpenAI):
        self.fixer_factory = FixerFactory(llm)
    
    async def apply_targeted_fixes(
        self,
        html_content: str,
        issues: List[ValidationIssue],
        spec: WhitePageSpec
    ) -> str:
        fixed_html = html_content
        
        sorted_issues = sorted(
            issues, 
            key=lambda x: (x.severity.value, x.issue_type.value)
        )
        
        for issue in sorted_issues:
            try:
                fixer = self.fixer_factory.get_fixer(issue.issue_type)
                fixed_html = await fixer.fix(fixed_html, issue, spec)
                logger.info(f"Applied fix for {issue.issue_type.value}: {issue.description[:50]}...")
            except Exception as e:
                logger.error(f"Failed to apply fix for {issue.issue_type.value}: {e}")
        
        return fixed_html


class TargetedFixingNode:
    """Main node for targeted fixing in the pipeline."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.targeted_html_fixer = TargetedHTMLFixer(llm)

    async def targeted_fixing_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing targeted fixing node for {state['brand_name']}")

        final_html = state.get("final_html")
        final_validation = state.get("final_validation")

        if not final_html or not final_validation:
            error_msg = "Missing HTML content or validation result for targeted fixing."
            logger.error(error_msg)
            return self._create_error_response(state, error_msg)

        try:
            spec = state["spec"]
            analyzer = ValidationAnalyzer(spec)
            issues = analyzer.analyze_validation_errors(final_validation, final_html)

            if not issues:
                logger.info("No fixable validation issues identified by analyzer.")
                return self._create_success_response(state, "No fixable issues found")

            fixed_html = await self.targeted_html_fixer.apply_targeted_fixes(
                final_html, issues, spec
            )

            metrics = state.get("metrics", {})
            metrics["targeted_fixes_applied"] = metrics.get("targeted_fixes_applied", 0) + len(issues)

            return {
                **state,
                "final_html": fixed_html,
                "metrics": metrics,
                "should_proceed_to_uniqueness": True,
                "should_apply_targeted_fixes": False,
                "messages": state.get("messages", []) + [
                    {"type": "info", "step": "targeted_fixing", 
                     "message": f"Applied {len(issues)} targeted fixes, proceeding to uniqueness"}
                ]
            }

        except Exception as e:
            error_msg = f"Error in targeted fixing node: {e}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(state, error_msg)

    def _create_error_response(self, state: GraphState, error_msg: str) -> Dict[str, Any]:
        return {
            **state,
            "should_retry_pipeline": True,
            "should_apply_targeted_fixes": False,
            "messages": state.get("messages", []) + [
                {"type": "error", "step": "targeted_fixing", "message": error_msg}
            ]
        }

    def _create_success_response(self, state: GraphState, message: str) -> Dict[str, Any]:
        return {
            **state,
            "should_proceed_to_uniqueness": True,
            "should_apply_targeted_fixes": False,
            "messages": state.get("messages", []) + [
                {"type": "info", "step": "targeted_fixing", "message": message}
            ]
        }