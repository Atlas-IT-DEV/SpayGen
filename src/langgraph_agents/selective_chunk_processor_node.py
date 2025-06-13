from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import logging
from bs4 import BeautifulSoup

from src.core.html_chunker import HTMLChunk, HTMLChunker, ChunkProcessingResult
from src.langgraph_agents.chunk_processor_node import ChunkProcessor
from src.models.pydantic_models import WhitePageSpec, GeneratedContent

from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ChunkValidationResult:
    chunk_id: str
    is_valid: bool
    score: float
    errors: List[str]
    warnings: List[str]
    needs_regeneration: bool = False

class ChunkContentValidator:
    def __init__(self, spec: WhitePageSpec):
        self.spec = spec

    def validate_chunk_content(self, chunk_content: str) -> Dict[str, Any]:
        errors = []
        warnings = []
        score = 1.0
        has_brand_mismatch = False

        brand_validation = self._validate_brand_presence(chunk_content)
        if not brand_validation["valid"]:
            errors.extend(brand_validation["errors"])
            score *= 0.7
            has_brand_mismatch = True

        contact_validation = self._validate_contact_info(chunk_content)
        if not contact_validation["valid"]:
            errors.extend(contact_validation["errors"])
            score *= 0.8

        content_validation = self._validate_content_relevance(chunk_content)
        if not content_validation["valid"]:
            warnings.extend(content_validation["warnings"])
            score *= 0.9

        return {
            "is_valid": score >= 0.7,
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "has_brand_mismatch": has_brand_mismatch
        }

    def _validate_brand_presence(self, content: str) -> Dict[str, Any]:
        content_lower = content.lower()
        brand_lower = self.spec.brand_name.lower()

        generic_brands = [
            "lumina jewelry", "stellar gems", "golden touch", "diamond dreams",
            "jewelry store", "company name", "brand name", "your brand"
        ]

        errors = []
        has_generic = any(generic in content_lower for generic in generic_brands)
        has_correct_brand = brand_lower in content_lower

        if has_generic and not has_correct_brand:
            errors.append(f"Contains generic brand names instead of '{self.spec.brand_name}'")
        elif has_generic:
            errors.append(f"Contains both correct and generic brand names")
        elif not has_correct_brand and any(word in content_lower for word in ["jewelry", "store", "brand"]):
            errors.append(f"Missing brand name '{self.spec.brand_name}' in relevant context")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _validate_contact_info(self, content: str) -> Dict[str, Any]:
        errors = []

        if "@" in content:
            if self.spec.contact_email not in content:
                if "example.com" in content or "gmail.com" in content:
                    errors.append(f"Contains generic email instead of '{self.spec.contact_email}'")

        phone_patterns = ["+7", "8-", "495", "123-45-67"] # These are examples, might need more robust regex
        if any(pattern in content for pattern in phone_patterns):
            if self.spec.contact_phone not in content:
                errors.append(f"Contains incorrect phone number instead of '{self.spec.contact_phone}'")

        # Basic address check (can be improved with more specific patterns)
        if "москва" in content.lower() or "moscow" in content.lower():
            if self.spec.address not in content:
                errors.append(f"Contains incorrect address instead of '{self.spec.address}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _validate_content_relevance(self, content: str) -> Dict[str, Any]:
        warnings = []
        content_lower = content.lower()

        if self.spec.products:
            product_terms = [product.lower() for product in self.spec.products]
            if not any(term in content_lower for term in product_terms):
                if any(word in content_lower for word in ["продукт", "товар", "украшение"]):
                    warnings.append("Contains generic product references instead of specific products")

        business_keywords = self.spec.business_description.lower().split()
        relevant_keywords = [word for word in business_keywords if len(word) > 4] # Filter short, common words

        if len(relevant_keywords) > 0:
            matching_keywords = sum(1 for keyword in relevant_keywords if keyword in content_lower)
            if matching_keywords / len(relevant_keywords) < 0.3: # If less than 30% of keywords match
                warnings.append("Content doesn't align well with business description")

        return {
            "valid": len(warnings) == 0,
            "warnings": warnings
        }

class SelectiveChunkProcessor:
    def __init__(self, chunk_processor: ChunkProcessor):
        self.chunk_processor = chunk_processor
        self.html_chunker = HTMLChunker() # HTMLChunker is stateless, can be instantiated here

    async def process_chunks_with_selective_retry(
        self,
        chunks: List[HTMLChunk],
        generated_content: GeneratedContent,
        spec: WhitePageSpec,
        max_retry_attempts: int = 2
    ) -> List[ChunkProcessingResult]:

        chunk_results = await self._initial_chunk_processing(chunks, generated_content, spec)

        for attempt in range(max_retry_attempts):
            validation_results = self._validate_chunk_results(chunk_results, spec)
            failed_chunk_ids = self._identify_failed_chunk_ids(validation_results)

            if not failed_chunk_ids:
                logger.info(f"All chunks passed validation on attempt {attempt + 1}")
                break

            logger.info(f"Retrying {len(failed_chunk_ids)} failed chunks (attempt {attempt + 1})")

            failed_chunks = [chunk for chunk in chunks if chunk.id in failed_chunk_ids]

            retry_results = await self._retry_failed_chunks(
                failed_chunks, generated_content, spec, attempt + 1
            )

            chunk_results = self._merge_chunk_results(chunk_results, retry_results)
        
        return chunk_results

    async def _initial_chunk_processing(
        self,
        chunks: List[HTMLChunk],
        generated_content: GeneratedContent,
        spec: WhitePageSpec
    ) -> List[ChunkProcessingResult]:
        return await self.chunk_processor.process_chunks_parallel(
            chunks, generated_content, spec
        )

    def _validate_chunk_results(
        self,
        chunk_results: List[ChunkProcessingResult],
        spec: WhitePageSpec
    ) -> List[ChunkValidationResult]:
        validation_results = []
        for result in chunk_results:
            validator = ChunkContentValidator(spec)
            validation = validator.validate_chunk_content(result.processed_content)
            validation_results.append(ChunkValidationResult(
                chunk_id=result.chunk_id,
                is_valid=validation["is_valid"],
                score=validation["score"],
                errors=validation["errors"],
                warnings=validation["warnings"],
                needs_regeneration=validation["score"] < 0.7 or validation.get("has_brand_mismatch", False)
            ))
        return validation_results

    def _identify_failed_chunk_ids(self, validation_results: List[ChunkValidationResult]) -> Set[str]:
        return {
            validation.chunk_id
            for validation in validation_results
            if validation.needs_regeneration
        }

    async def _retry_failed_chunks(
        self,
        failed_chunks: List[HTMLChunk],
        generated_content: GeneratedContent,
        spec: WhitePageSpec,
        retry_attempt: int
    ) -> List[ChunkProcessingResult]:
        enhanced_content = self._enhance_content_for_retry(generated_content, spec, retry_attempt)
        
        retry_results = []
        for chunk in failed_chunks:
            logger.debug(f"Retrying chunk {chunk.id} (attempt {retry_attempt})")
            result = await self.chunk_processor.process_chunk_with_enhanced_instructions(
                chunk, enhanced_content, spec, retry_attempt
            )
            retry_results.append(result)
        return retry_results

    def _enhance_content_for_retry(
        self,
        generated_content: GeneratedContent,
        spec: WhitePageSpec,
        retry_attempt: int
    ) -> GeneratedContent:
        enhanced_content = generated_content.model_copy(deep=True)
        if retry_attempt == 1:
            enhanced_content.enhancement_instructions = [
                f"CRITICAL: Use exact brand name '{spec.brand_name}' - no variations allowed",
                f"CRITICAL: Use exact contact email '{spec.contact_email}'",
                f"CRITICAL: Use exact phone number '{spec.contact_phone}'",
                f"CRITICAL: Use exact address '{spec.address}'"
            ]
        elif retry_attempt == 2:
            enhanced_content.enhancement_instructions = [
                f"MANDATORY REPLACEMENT: Replace ANY occurrence of brand names with '{spec.brand_name}'",
                f"MANDATORY REPLACEMENT: Replace ALL email addresses with '{spec.contact_email}'",
                f"MANDATORY REPLACEMENT: Replace ALL phone numbers with '{spec.contact_phone}'",
                f"MANDATORY REPLACEMENT: Replace ALL addresses with '{spec.address}'",
                "SEARCH AND REPLACE: Look for common generic terms and replace with specific content"
            ]
        return enhanced_content

    def _merge_chunk_results(
        self,
        original_results: List[ChunkProcessingResult],
        retry_results: List[ChunkProcessingResult]
    ) -> List[ChunkProcessingResult]:
        retry_dict = {result.chunk_id: result for result in retry_results}
        merged_results = []
        for original in original_results:
            if original.chunk_id in retry_dict:
                merged_results.append(retry_dict[original.chunk_id])
            else:
                merged_results.append(original)
        return merged_results