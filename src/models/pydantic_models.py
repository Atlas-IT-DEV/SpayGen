from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class PageTypeEnum(str, Enum):
    ECOMMERCE = "ecommerce"
    JEWELRY = "jewelry"
    EDUCATION = "education"
    MOUNTAINEERING = "mountaineering"
    # Add other page types as needed

class WhitePageSpec(BaseModel):
    page_type: PageTypeEnum = Field(..., description="Тип страницы")
    brand_name: str = Field(..., description="Название бренда")
    business_description: str = Field(..., description="Описание бизнеса")
    contact_email: str = Field(..., description="Email для контактов")
    contact_phone: str = Field(..., description="Телефон для контактов")
    address: str = Field(..., description="Физический адрес")
    page_name: str = Field(..., description="Имя файла для сохранения страницы (без расширения)")
    custom_button_label: Optional[str] = Field(None, description="Кастомная кнопка")
    products: Optional[List[str]] = Field(None, description="Список товаров")
    geo_location: str = Field("USA", description="Географическое расположение")
    page_description: Optional[str] = Field(None, description="Текстовое описание желаемого вида и содержания страницы")

    @validator('page_name')
    def validate_page_name(cls, v):
        if not v or not v.strip():
            raise ValueError('page_name cannot be empty')
        return v.strip()

class FullPageTemplate(BaseModel):
    name: str = Field(..., description="Название полностраничного шаблона")
    html: str = Field(..., description="Полный HTML код страницы")
    css: str = Field("", description="Полные CSS стили страницы")
    description: str = Field(..., description="Текстовое описание шаблона для поиска")
    tags: List[str] = Field(default_factory=list, description="Теги для поиска и категоризации шаблона")

class GeneratedContent(BaseModel):
    main_content: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key-value pairs for main text content"
    )
    items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of structured items (products, services, team members)"
    )
    images: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key-value pairs for image URLs or descriptions"
    )
    contact_info: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key-value pairs for contact information"
    )
    other_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Any other specific data needed for template modification"
    )
    enhancement_instructions: Optional[List[str]] = Field(
        default_factory=list,
        description="Инструкции для улучшения"
    )

class ValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Результат валидации")
    errors: List[str] = Field(default_factory=list, description="Список ошибок")
    warnings: List[str] = Field(default_factory=list, description="Предупреждения")
    score: float = Field(0.0, description="Оценка качества страницы")

class GeneratedWhitePage(BaseModel):
    html: str = Field(..., description="Сгенерированный HTML")
    css: str = Field("", description="CSS стили")
    spec: WhitePageSpec = Field(..., description="Спецификация страницы")
    validation: ValidationResult = Field(..., description="Результат валидации")
    generation_time: datetime = Field(default_factory=datetime.now)

class WhitePageRequest(BaseModel):
    spec: WhitePageSpec = Field(..., description="Спецификация для генерации")
    template_preferences: Optional[Dict[str, str]] = Field(None, description="Предпочтения шаблонов")

class WhitePageResponse(BaseModel):
    success: bool = Field(..., description="Успешность генерации")
    generated_page: Optional[GeneratedWhitePage] = Field(None, description="Сгенерированная страница")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    preview_url: Optional[str] = Field(None, description="URL для предпросмотра страницы")
    generation_time: Optional[float] = Field(None, description="Время генерации в секундах")