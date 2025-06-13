from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class HTMLChunk:
    id: str
    content: str
    chunk_type: str
    dependencies: List[str]
    css_selectors: List[str]
    size: int
    metadata: Dict[str, Any]


@dataclass
class ChunkProcessingResult:
    chunk_id: str
    processed_content: str
    validation_score: float
    errors: List[str]
    warnings: List[str]


class HTMLChunker:
    MAX_CHUNK_SIZE = 8000
    STRUCTURAL_TAGS = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer']
    
    def __init__(self, max_chunk_size: int = None):
        self.max_chunk_size = max_chunk_size or self.MAX_CHUNK_SIZE
    
    def split_html(self, html_content: str, css_content: str = "") -> List[HTMLChunk]:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        head_chunk = self._create_head_chunk(soup)
        body_chunks = self._split_body_content(soup)
        css_classes = self._extract_css_classes(css_content)
        
        chunks = [head_chunk] + body_chunks
        
        for chunk in chunks:
            chunk.css_selectors = self._find_relevant_css_selectors(chunk.content, css_classes)
        
        return chunks
    
    def _create_head_chunk(self, soup: BeautifulSoup) -> HTMLChunk:
        head = soup.find('head')
        head_content = str(head) if head else "<head></head>"
        
        return HTMLChunk(
            id="head",
            content=head_content,
            chunk_type="head",
            dependencies=[],
            css_selectors=[],
            size=len(head_content),
            metadata={"priority": "high", "modifiable": False}
        )
    
    def _split_body_content(self, soup: BeautifulSoup) -> List[HTMLChunk]:
        body = soup.find('body')
        if not body:
            return []
        
        chunks = []
        structural_elements = self._find_structural_elements(body)
        
        for i, element in enumerate(structural_elements):
            chunk_content = str(element)
            
            if len(chunk_content) > self.max_chunk_size:
                sub_chunks = self._split_large_element(element, i)
                chunks.extend(sub_chunks)
            else:
                chunk = HTMLChunk(
                    id=f"section_{i}",
                    content=chunk_content,
                    chunk_type=element.name or "div",
                    dependencies=[],
                    css_selectors=[],
                    size=len(chunk_content),
                    metadata=self._extract_element_metadata(element)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _find_structural_elements(self, body: Tag) -> List[Tag]:
        structural_elements = []
        
        for tag in self.STRUCTURAL_TAGS:
            elements = body.find_all(tag, recursive=False)
            structural_elements.extend(elements)
        
        if not structural_elements:
            direct_children = [child for child in body.children if hasattr(child, 'name')]
            structural_elements = direct_children[:10]
        
        return structural_elements
    
    def _split_large_element(self, element: Tag, section_index: int) -> List[HTMLChunk]:
        chunks = []
        child_elements = [child for child in element.children if hasattr(child, 'name')]
        
        current_chunk_content = []
        current_size = 0
        chunk_counter = 0
        
        opening_tag = self._get_opening_tag(element)
        closing_tag = f"</{element.name}>"
        base_size = len(opening_tag) + len(closing_tag)
        
        for child in child_elements:
            child_str = str(child)
            child_size = len(child_str)
            
            if current_size + child_size + base_size > self.max_chunk_size and current_chunk_content:
                chunk_content = opening_tag + ''.join(current_chunk_content) + closing_tag
                
                chunk = HTMLChunk(
                    id=f"section_{section_index}_{chunk_counter}",
                    content=chunk_content,
                    chunk_type=f"{element.name}_part",
                    dependencies=[],
                    css_selectors=[],
                    size=len(chunk_content),
                    metadata=self._extract_element_metadata(element)
                )
                chunks.append(chunk)
                
                current_chunk_content = []
                current_size = 0
                chunk_counter += 1
            
            current_chunk_content.append(child_str)
            current_size += child_size
        
        if current_chunk_content:
            chunk_content = opening_tag + ''.join(current_chunk_content) + closing_tag
            
            chunk = HTMLChunk(
                id=f"section_{section_index}_{chunk_counter}",
                content=chunk_content,
                chunk_type=f"{element.name}_part",
                dependencies=[],
                css_selectors=[],
                size=len(chunk_content),
                metadata=self._extract_element_metadata(element)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_opening_tag(self, element: Tag) -> str:
        attrs = []
        for key, value in element.attrs.items():
            if isinstance(value, list):
                value = ' '.join(value)
            attrs.append(f'{key}="{value}"')
        
        attrs_str = ' ' + ' '.join(attrs) if attrs else ''
        return f"<{element.name}{attrs_str}>"
    
    def _extract_element_metadata(self, element: Tag) -> Dict[str, Any]:
        metadata = {
            "tag_name": element.name,
            "classes": element.get('class', []),
            "id": element.get('id'),
            "priority": "medium",
            "modifiable": True
        }
        
        if element.name in ['nav', 'footer']:
            metadata["priority"] = "low"
            metadata["modifiable"] = False
        elif element.name in ['header', 'main']:
            metadata["priority"] = "high"
        
        return metadata
    
    def _extract_css_classes(self, css_content: str) -> List[str]:
        if not css_content:
            return []
        
        class_pattern = r'\.([a-zA-Z_-][a-zA-Z0-9_-]*)'
        classes = re.findall(class_pattern, css_content)
        return list(set(classes))
    
    def _find_relevant_css_selectors(self, chunk_content: str, css_classes: List[str]) -> List[str]:
        relevant_selectors = []
        
        for css_class in css_classes:
            if f'class="{css_class}"' in chunk_content or f"class='{css_class}'" in chunk_content:
                relevant_selectors.append(f".{css_class}")
        
        return relevant_selectors
    
    def reconstruct_html(self, processed_chunks: List[ChunkProcessingResult], original_doctype: str = "") -> str:
        if not processed_chunks:
            return ""
        
        head_chunk = next((chunk for chunk in processed_chunks if chunk.chunk_id == "head"), None)
        body_chunks = [chunk for chunk in processed_chunks if chunk.chunk_id != "head"]
        
        head_content = head_chunk.processed_content if head_chunk else "<head></head>"
        body_content = ''.join([chunk.processed_content for chunk in body_chunks])
        
        doctype = original_doctype or "<!DOCTYPE html>"
        
        return f"{doctype}\n<html>\n{head_content}\n<body>\n{body_content}\n</body>\n</html>"