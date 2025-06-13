from typing import Dict, List, Any
from bs4 import BeautifulSoup
import re
from src.models.pydantic_models import ValidationResult
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class BingValidator:
    """
    Класс для валидации HTML-страниц на соответствие критериям Bing Ads.
    """
    def __init__(self) -> None:
        self.required_policies = settings.bing_required_policies
        self.forbidden_keywords = settings.bing_forbidden_keywords

    def validate(self, html: str, spec: Dict[str, Any] = None) -> ValidationResult:
        """
        Выполняет валидацию HTML-страницы.
        Args:
            html: HTML-контент страницы.
            spec: Спецификация страницы, содержащая контактную информацию.
        Returns:
            ValidationResult: Результат валидации, включая ошибки, предупреждения и оценку.
        """
        soup = BeautifulSoup(html, 'html.parser')
        errors = []
        warnings = []
        score = 1.0 # Начинаем со 100%

        # Проверки, которые всегда добавляют ошибку, если не могут быть выполнены
        # (например, проверка SSL, которая должна быть на уровне HTTPS)
        # errors.extend(self._check_ssl_requirement()) # Удалено, так как это не может быть проверено здесь

        errors.extend(self._check_required_policies(soup))

        if spec:
            errors.extend(self._check_contact_info(soup, spec))

        errors.extend(self._check_forbidden_content(soup))
        errors.extend(self._check_commerce_functionality(soup))
        warnings.extend(self._check_content_quality(soup))

        # Расчет оценки
        if errors:
            score -= len(errors) * 0.15 # Каждая ошибка снижает оценку на 15%
        if warnings:
            score -= len(warnings) * 0.05 # Каждое предупреждение снижает оценку на 5%

        score = max(0.0, min(1.0, score)) # Оценка от 0.0 до 1.0

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            score=score
        )

    # Метод validate_page был дубликатом validate, поэтому удален.
    # def validate_page(self, html: str, spec: Dict[str, Any]) -> ValidationResult:
    #     return self.validate(html, spec)

    # _check_ssl_requirement удален, так как его проверка невозможна на этом уровне.
    # def _check_ssl_requirement(self) -> List[str]:
    #     return ["SSL certificate is required for HTTPS"]

    def _check_required_policies(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет наличие обязательных политик на странице.
        """
        errors = []
        text_content = soup.get_text().lower()

        for policy in self.required_policies:
            if policy not in text_content:
                errors.append(f"Missing required policy: {policy}")

        return errors

    def _check_contact_info(self, soup: BeautifulSoup, spec: Dict[str, Any]) -> List[str]:
        """
        Проверяет наличие и корректность контактной информации на странице.
        """
        errors = []
        text_content = soup.get_text()

        contact_phone = spec.get("contact_phone", "")
        contact_email = spec.get("contact_email", "")
        address = spec.get("address", "")

        if not contact_phone or contact_phone not in text_content:
            errors.append("Valid contact phone number is required")

        if not contact_email or contact_email not in text_content:
            errors.append("Valid contact email is required")

        if not address or len(address) < 10: # Простая проверка на минимальную длину адреса
            errors.append("Detailed physical address is required")

        return errors

    def _check_forbidden_content(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет наличие запрещенных ключевых слов на странице.
        """
        errors = []
        text_content = soup.get_text().lower()

        for keyword in self.forbidden_keywords:
            if keyword in text_content:
                errors.append(f"Forbidden content detected: {keyword}")

        return errors

    def _check_commerce_functionality(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет наличие базовой функциональности электронной коммерции (кнопки, формы).
        """
        errors = []

        # Проверка наличия кнопок "добавить в корзину" / "купить"
        if not soup.find_all("button", string=re.compile(r"add to cart|buy|order|заказать|купить", re.I)):
            errors.append("E-commerce functionality (cart/buy buttons) is required")

        # Проверка наличия форм (контактных, заказа и т.д.)
        if not soup.find("form"):
            errors.append("Contact/order forms are required")

        return errors

    def _check_content_quality(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет качество контента страницы (длина текста, количество изображений).
        """
        warnings = []
        text_content = soup.get_text()

        if len(text_content) < 500:
            warnings.append("Page content is too short")

        if len(soup.find_all("img")) < 3:
            warnings.append("Consider adding more product images")

        return warnings