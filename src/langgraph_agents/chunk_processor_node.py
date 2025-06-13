from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import logging
from bs4 import BeautifulSoup

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.core.html_chunker import HTMLChunk, ChunkProcessingResult
from src.models.pydantic_models import WhitePageSpec, GeneratedContent
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ChunkModificationInput:
    chunk: HTMLChunk
    generated_content: GeneratedContent
    spec: WhitePageSpec
    context: Dict[str, Any]

class ChunkProcessor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.str_parser = StrOutputParser()

    async def process_chunk(
        self,
        chunk: HTMLChunk,
        generated_content: GeneratedContent,
        spec: WhitePageSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> ChunkProcessingResult:
        if not chunk.metadata.get("modifiable", True):
            return ChunkProcessingResult(
                chunk_id=chunk.id,
                processed_content=chunk.content,
                validation_score=1.0,
                errors=[],
                warnings=["Chunk marked as non-modifiable"]
            )

        try:
            modification_input = ChunkModificationInput(
                chunk=chunk,
                generated_content=generated_content,
                spec=spec,
                context=context or {}
            )

            processed_content = await self._modify_chunk_content(modification_input)
            validation_result = self._validate_chunk(processed_content, chunk)

            return ChunkProcessingResult(
                chunk_id=chunk.id,
                processed_content=processed_content,
                validation_score=validation_result["score"],
                errors=validation_result["errors"],
                warnings=validation_result["warnings"]
            )

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk.id}: {e}")
            return ChunkProcessingResult(
                chunk_id=chunk.id,
                processed_content=chunk.content,
                validation_score=0.0,
                errors=[str(e)],
                warnings=[]
            )

    async def process_chunks_parallel(
        self,
        chunks: List[HTMLChunk],
        generated_content: GeneratedContent,
        spec: WhitePageSpec
    ) -> List[ChunkProcessingResult]:
        semaphore = asyncio.Semaphore(3)

        async def process_single_chunk(chunk: HTMLChunk) -> ChunkProcessingResult:
            async with semaphore:
                return await self.process_chunk(chunk, generated_content, spec)

        tasks = [process_single_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {chunks[i].id} processing failed: {result}")
                processed_results.append(ChunkProcessingResult(
                    chunk_id=chunks[i].id,
                    processed_content=chunks[i].content,
                    validation_score=0.0,
                    errors=[str(result)],
                    warnings=[]
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def process_chunk_with_enhanced_instructions(
        self,
        chunk: HTMLChunk,
        enhanced_content: GeneratedContent,
        spec: WhitePageSpec,
        retry_attempt: int
    ) -> ChunkProcessingResult:
        try:
            if not chunk.metadata.get("modifiable", True):
                return self._create_unmodified_result(chunk)

            enhanced_prompt_template = self._create_enhanced_chunk_prompt_template(
                chunk, enhanced_content, spec, retry_attempt
            )

            processed_content = await self._process_with_enhanced_llm(enhanced_prompt_template, chunk, enhanced_content, spec)
            validation_result = self._validate_chunk(processed_content, chunk)

            return ChunkProcessingResult(
                chunk_id=chunk.id,
                processed_content=processed_content,
                validation_score=validation_result["score"],
                errors=validation_result["errors"],
                warnings=validation_result["warnings"]
            )

        except Exception as e:
            logger.error(f"Enhanced processing failed for chunk {chunk.id}: {e}")
            return self._create_error_result(chunk, str(e))

    async def _modify_chunk_content(self, modification_input: ChunkModificationInput) -> str:
        chunk = modification_input.chunk
        generated_content = modification_input.generated_content
        spec = modification_input.spec

        if chunk.chunk_type == "head":
            return self._modify_head_chunk(chunk, spec)

        prompt_template = self._create_chunk_modification_prompt_template(chunk, generated_content, spec)
        chain = prompt_template | self.llm | self.str_parser

        llm_output = await chain.ainvoke({
            "chunk_content": chunk.content,
            "brand_name": spec.brand_name,
            "products": generated_content.items,
            "images": generated_content.images,
            "contact": generated_content.contact_info
        })
        
        return self._extract_modified_content(llm_output, chunk.content)

    async def _process_with_enhanced_llm(
        self, 
        enhanced_prompt_template: ChatPromptTemplate, 
        chunk: HTMLChunk,
        enhanced_content: GeneratedContent,
        spec: WhitePageSpec
    ) -> str:
        chain = enhanced_prompt_template | self.llm | self.str_parser
        
        llm_output = await chain.ainvoke({
            "chunk_content": chunk.content,
            "brand_name": spec.brand_name,
            "products": enhanced_content.items,
            "images": enhanced_content.images,
            "contact": enhanced_content.contact_info
        })
        
        return self._extract_modified_content(llm_output, chunk.content)

    def _create_chunk_modification_prompt_template(
        self,
        chunk: HTMLChunk,
        generated_content: GeneratedContent,
        spec: WhitePageSpec
    ) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
            Modify this HTML chunk by replacing placeholder content with real data.

            CRITICAL RULES:
            1. Preserve the exact HTML structure and all attributes
            2. Only replace text content and image URLs (src attributes)
            3. Keep all CSS classes and IDs unchanged
            4. Return only the modified HTML chunk, nothing else. Do NOT include any markdown or extra text.
            """),
            ("human", """
            Original chunk:
            ```html
            {chunk_content}
            ```

            Available content for replacement:
            - Brand: {brand_name}
            - Products: {products}
            - Images: {images}
            - Contact: {contact}

            Return ONLY the modified HTML chunk:
            """)
        ])

    def _create_enhanced_chunk_prompt_template(
        self,
        chunk: HTMLChunk,
        enhanced_content: GeneratedContent,
        spec: WhitePageSpec,
        retry_attempt: int
    ) -> ChatPromptTemplate:
        base_system_message = """
        Modify this HTML chunk by replacing placeholder content with real data.

        CRITICAL RULES:
        1. Preserve the exact HTML structure and all attributes
        2. Only replace text content and image URLs (src attributes)
        3. Keep all CSS classes and IDs unchanged
        4. Return only the modified HTML chunk, nothing else. Do NOT include any markdown or extra text.
        """
        
        enhancement_instructions = getattr(enhanced_content, 'enhancement_instructions', [])
        enhancement_text = "\n".join(enhancement_instructions) if enhancement_instructions else ""

        if retry_attempt >= 2:
            strict_instructions = f"""
    CRITICAL MANDATORY REPLACEMENTS - NO EXCEPTIONS:
    1. Find and replace ANY brand name with: {spec.brand_name}
    2. Find and replace ANY email with: {spec.contact_email}
    3. Find and replace ANY phone with: {spec.contact_phone}
    4. Find and replace ANY address with: {spec.address}

    SEARCH PATTERNS TO REPLACE:
    - "Lumina Jewelry" -> "{spec.brand_name}"
    - "Stellar Gems" -> "{spec.brand_name}"
    - "Golden Touch" -> "{spec.brand_name}"
    - Any email ending in @example.com, @gmail.com -> {spec.contact_email}
    - Any phone like +1, 555-, (123) -> {spec.contact_phone}
    - Any address mentioning other cities -> {spec.address}

    THIS IS ATTEMPT {retry_attempt} - PREVIOUS ATTEMPTS FAILED DUE TO INCORRECT CONTENT.
    """
        else:
            strict_instructions = f"""
    IMPORTANT: This is retry attempt {retry_attempt}. Previous attempt failed validation.
    Focus on using EXACT values from spec:
    - Brand: {spec.brand_name}
    - Email: {spec.contact_email}
    - Phone: {spec.contact_phone}
    - Address: {spec.address}
    """
        
        system_message = base_system_message + "\n\n" + strict_instructions + "\n\n" + enhancement_text
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", """
            Original chunk:
            ```html
            {chunk_content}
            ```

            Available content for replacement:
            - Brand: {brand_name}
            - Products: {products}
            - Images: {images}
            - Contact: {contact}

            Return ONLY the modified HTML chunk:
            """)
        ])

    def _modify_head_chunk(self, chunk: HTMLChunk, spec: WhitePageSpec) -> str:
        soup = BeautifulSoup(chunk.content, 'html.parser')

        title = soup.find('title')
        if title:
            title.string = f"{spec.brand_name} - {spec.business_description[:50]}..."

        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            meta_description['content'] = spec.business_description

        return str(soup)

    def _extract_modified_content(self, llm_output: str, fallback_content: str) -> str:
        cleaned_output = llm_output.strip()
        if cleaned_output.startswith('<') and cleaned_output.endswith('>'):
            return cleaned_output
        logger.warning(f"LLM output for chunk modification did not look like HTML. Falling back. Output: {cleaned_output[:200]}...")
        return fallback_content

    def _validate_chunk(self, processed_content: str, original_chunk: HTMLChunk) -> Dict[str, Any]:
        errors = []
        warnings = []
        score = 1.0

        if not processed_content.strip():
            errors.append("Processed content is empty")
            score = 0.0

        if len(processed_content) < len(original_chunk.content) * 0.5:
            warnings.append("Significant content reduction detected")
            score *= 0.8

        try:
            soup = BeautifulSoup(processed_content, 'html.parser')
            if not soup.find():
                errors.append("Invalid HTML structure (no tags found)")
                score = 0.0
        except Exception as e:
            errors.append(f"HTML parsing error: {e}")
            score = 0.0

        return {
            "score": score,
            "errors": errors,
            "warnings": warnings
        }

    def _create_unmodified_result(self, chunk: HTMLChunk) -> ChunkProcessingResult:
        return ChunkProcessingResult(
            chunk_id=chunk.id,
            processed_content=chunk.content,
            validation_score=1.0,
            errors=[],
            warnings=["Chunk marked as non-modifiable"]
        )

    def _create_error_result(self, chunk: HTMLChunk, error_msg: str) -> ChunkProcessingResult:
        return ChunkProcessingResult(
            chunk_id=chunk.id,
            processed_content=chunk.content,
            validation_score=0.0,
            errors=[error_msg],
            warnings=[]
        )