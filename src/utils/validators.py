from typing import Dict, List, Any, Optional, Tuple
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import dns.resolver
import socket
from email_validator import validate_email, EmailNotValidError
from src.config.settings import settings # Импортируем настройки

class DataValidator:
    """
    Набор статических методов для валидации различных типов данных.
    """
    @staticmethod
    def validate_email_address(email: str) -> Tuple[bool, str]:
        """
        Валидирует формат email-адреса.
        """
        try:
            validated_email = validate_email(email)
            return True, validated_email.email
        except EmailNotValidError as e:
            return False, str(e)

    @staticmethod
    def validate_phone_number(phone: str) -> Tuple[bool, str]:
        """
        Валидирует формат телефонного номера (простой паттерн для США).
        """
        phone_pattern = re.compile(r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$')

        if phone_pattern.match(phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')):
            return True, phone
        else:
            return False, "Invalid phone number format"

    @staticmethod
    def validate_address(address: str) -> Tuple[bool, str]:
        """
        Валидирует адрес на минимальную длину и наличие цифр/букв.
        """
        if len(address) < 10:
            return False, "Address too short"

        has_numbers = bool(re.search(r'\d', address))
        has_letters = bool(re.search(r'[a-zA-Z]', address))

        if not (has_numbers and has_letters):
            return False, "Address must contain both numbers and letters"

        return True, address

    @staticmethod
    def validate_brand_name(brand_name: str) -> Tuple[bool, str]:
        """
        Валидирует название бренда на длину и наличие запрещенных слов.
        """
        if len(brand_name) < 2:
            return False, "Brand name too short"

        if len(brand_name) > 50:
            return False, "Brand name too long"

        forbidden_words = ['test', 'example', 'sample', 'demo']
        if any(word in brand_name.lower() for word in forbidden_words):
            return False, "Brand name contains forbidden words"

        return True, brand_name

    @staticmethod
    def validate_business_description(description: str) -> Tuple[bool, str]:
        """
        Валидирует описание бизнеса на длину.
        """
        if len(description) < 20:
            return False, "Business description too short"

        if len(description) > 500:
            return False, "Business description too long"

        return True, description

class HTMLValidator:
    """
    Класс для валидации структуры и содержимого HTML.
    """
    def __init__(self):
        self.required_tags = settings.html_required_tags
        self.required_meta_tags = settings.html_required_meta_tags
        self.forbidden_tags = settings.html_forbidden_tags

    def validate_html_structure(self, html: str) -> Tuple[bool, List[str]]:
        """
        Валидирует базовую HTML-структуру, наличие мета-тегов и отсутствие запрещенных тегов.
        """
        errors = []

        try:
            soup = BeautifulSoup(html, 'html.parser')
        except Exception as e:
            return False, [f"Invalid HTML structure: {str(e)}"]

        errors.extend(self._check_required_tags(soup))
        errors.extend(self._check_meta_tags(soup))
        errors.extend(self._check_forbidden_content(soup))
        errors.extend(self._check_forms(soup))
        errors.extend(self._check_links(soup))

        return len(errors) == 0, errors

    def validate_css(self, css: str) -> Tuple[bool, List[str]]:
        """
        Валидирует CSS-контент на наличие запрещенных свойств.
        """
        errors = []

        if not css.strip():
            errors.append("No CSS content provided")
            return False, errors

        for prop in settings.css_forbidden_properties:
            if prop in css:
                errors.append(f"Forbidden CSS property: {prop}")

        return len(errors) == 0, errors

    def _check_required_tags(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет наличие обязательных HTML-тегов.
        """
        errors = []
        for tag in self.required_tags:
            if not soup.find(tag):
                errors.append(f"Missing required tag: {tag}")
        return errors

    def _check_meta_tags(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет наличие обязательных мета-тегов.
        """
        errors = []
        meta_tags = soup.find_all('meta')
        found_meta = set()

        for meta in meta_tags:
            if meta.get('charset'):
                found_meta.add('charset')
            if meta.get('name') == 'viewport':
                found_meta.add('viewport')

        for required_meta in self.required_meta_tags:
            if required_meta not in found_meta:
                errors.append(f"Missing required meta tag: {required_meta}")
        return errors

    def _check_forbidden_content(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет наличие запрещенных HTML-тегов.
        """
        errors = []
        for tag in self.forbidden_tags:
            if soup.find(tag):
                errors.append(f"Forbidden tag found: {tag}")
        return errors

    def _check_forms(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет наличие форм и их базовых атрибутов.
        """
        errors = []
        forms = soup.find_all('form')
        if not forms:
            errors.append("No forms found - required for e-commerce")
            return errors

        for form in forms:
            if not form.get('action'):
                errors.append("Form missing action attribute")
            if not form.get('method'):
                errors.append("Form missing method attribute")
            required_inputs = form.find_all(['input', 'textarea'])
            if not required_inputs:
                errors.append("Form has no input fields")
        return errors

    def _check_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Проверяет валидность внешних URL-адресов в ссылках.
        """
        errors = []
        links = soup.find_all('a')
        for link in links:
            href = link.get('href')
            if href and href.startswith('http'):
                if not self._is_valid_url(href):
                    errors.append(f"Invalid external URL: {href}")
        return errors

    def _is_valid_url(self, url: str) -> bool:
        """
        Проверяет, является ли URL валидным.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

class SecurityValidator:
    """
    Класс для валидации безопасности входных данных.
    """
    @staticmethod
    def validate_input_security(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Проверяет входные данные на наличие потенциально опасного контента (XSS).
        """
        errors = []
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'onload=',
            r'onerror=',
            r'eval\(',
            r'document\.cookie',
            r'window\.location'
        ]

        for key, value in data.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Potentially dangerous content in {key}")
        return len(errors) == 0, errors