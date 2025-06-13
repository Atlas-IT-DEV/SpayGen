from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import asyncio
import sys

from langchain_openai import OpenAIEmbeddings

from src.models.pydantic_models import FullPageTemplate, WhitePageSpec
from src.tools.qdrant_manager import AsyncQdrantManager
from src.config.settings import settings

logger = logging.getLogger(__name__)

class TemplateLoader:
    def __init__(self, qdrant_manager: AsyncQdrantManager, templates_dir: str = "templates/full_page_html"):
        self.templates_dir = Path(templates_dir)
        self.qdrant = qdrant_manager
        self.embedding_model = OpenAIEmbeddings(
            model=settings.model_settings.get("embedding_model", "text-embedding-3-small"),
            openai_api_key=settings.openai_api_key
        )
        self.collection_name = "full_page_templates"

        print(f"[INIT] TemplateLoader initialized with dir: {self.templates_dir}")
        logger.info("TemplateLoader initialized successfully")

    async def load_all_templates(self) -> None:
        print("[TEMPLATE_LOADER] Starting load_all_templates method")
        print(f"[TEMPLATE_LOADER] Templates dir: {self.templates_dir}")
        print(f"[TEMPLATE_LOADER] Working dir: {Path.cwd()}")

        try:
            print("[TEMPLATE_LOADER] About to log with logger")
            logger.info("=" * 60)
            logger.info("STARTING TEMPLATE LOADING PROCESS")
            logger.info("=" * 60)

            print(f"[TEMPLATE_LOADER] Directory exists: {self.templates_dir.exists()}")
            logger.info(f"Templates directory: {self.templates_dir.absolute()}")
            logger.info(f"Directory exists: {self.templates_dir.exists()}")

            if not self.templates_dir.exists():
                print("[TEMPLATE_LOADER] Creating directory...")
                logger.warning(f"Creating templates directory: {self.templates_dir}")
                self.templates_dir.mkdir(parents=True, exist_ok=True)
                print("[TEMPLATE_LOADER] Directory created")

            print("[TEMPLATE_LOADER] Listing directory contents...")
            items = list(self.templates_dir.iterdir())
            print(f"[TEMPLATE_LOADER] Found {len(items)} items")

            logger.info(f"Found {len(items)} items in templates directory:")
            for item in items:
                item_type = 'DIR' if item.is_dir() else 'FILE'
                print(f"[TEMPLATE_LOADER]   - {item.name} ({item_type})")
                logger.info(f"  - {item.name} ({item_type})")

            if not items:
                print("[TEMPLATE_LOADER] No items found - returning early")
                logger.warning("No items found in templates directory")
                return

            loaded_count = 0
            processed_count = 0

            print(f"[TEMPLATE_LOADER] Starting to process {len(items)} items...")

            for template_dir in items:
                processed_count += 1
                print(f"[TEMPLATE_LOADER] Processing {processed_count}/{len(items)}: {template_dir.name}")
                logger.info(f"Processing {processed_count}/{len(items)}: {template_dir.name}")

                if not template_dir.is_dir():
                    print(f"[TEMPLATE_LOADER] Skipping non-directory: {template_dir.name}")
                    logger.info(f"Skipping non-directory: {template_dir.name}")
                    continue

                html_file = template_dir / "template.html"
                css_file = template_dir / "style.css"
                description_file = template_dir / "description.txt"

                html_exists = html_file.exists()
                css_exists = css_file.exists()
                desc_exists = description_file.exists()

                print(f"[TEMPLATE_LOADER] Files in {template_dir.name}:")
                print(f"[TEMPLATE_LOADER]   template.html: {html_exists}")
                print(f"[TEMPLATE_LOADER]   style.css: {css_exists}")
                print(f"[TEMPLATE_LOADER]   description.txt: {desc_exists}")

                logger.info(f"Files in {template_dir.name}:")
                logger.info(f"  template.html: {html_exists}")
                logger.info(f"  style.css: {css_exists}")
                logger.info(f"  description.txt: {desc_exists}")

                if not html_exists:
                    print(f"[TEMPLATE_LOADER] Missing template.html in {template_dir.name} - SKIPPING")
                    logger.warning(f"Missing template.html in {template_dir.name} - SKIPPING")
                    continue

                template_name = template_dir.name
                print(f"[TEMPLATE_LOADER] Processing template: {template_name}")

                try:
                    print(f"[TEMPLATE_LOADER] Ensuring Qdrant collection...")
                    client = await self.qdrant.get_client()
                    await self.qdrant._ensure_collection(client, self.collection_name)
                    print(f"[TEMPLATE_LOADER] Collection ensured")

                    print(f"[TEMPLATE_LOADER] Checking if template exists...")
                    existing_template = await self.qdrant.get_template(template_name, collection_name=self.collection_name)

                    if existing_template:
                        print(f"[TEMPLATE_LOADER] Template '{template_name}' already exists - SKIPPING")
                        logger.info(f"Template '{template_name}' already exists - SKIPPING")
                        loaded_count += 1
                        continue

                    print(f"[TEMPLATE_LOADER] Loading template files...")
                    template = await self.load_full_page_template_from_files(
                        template_name, html_file, css_file, description_file
                    )

                    if template:
                        print(f"[TEMPLATE_LOADER] Saving template to Qdrant...")
                        await self.save_full_page_template_to_qdrant(template)
                        print(f"[TEMPLATE_LOADER] ✅ SUCCESS: Template '{template.name}' loaded and saved")
                        logger.info(f"✅ SUCCESS: Template '{template.name}' loaded and saved")
                        loaded_count += 1
                    else:
                        print(f"[TEMPLATE_LOADER] ❌ FAILED: Could not load template '{template_name}'")
                        logger.error(f"❌ FAILED: Could not load template '{template_name}'")

                except Exception as e:
                    print(f"[TEMPLATE_LOADER] ❌ ERROR processing template {template_name}: {e}")
                    logger.error(f"❌ ERROR processing template {template_name}: {e}", exc_info=True)
                    continue

            print(f"[TEMPLATE_LOADER] COMPLETED - Processed: {processed_count}, Loaded: {loaded_count}")
            logger.info("=" * 60)
            logger.info(f"TEMPLATE LOADING RESULTS:")
            logger.info(f"Processed: {processed_count}")
            logger.info(f"Loaded/Existing: {loaded_count}")
            logger.info("=" * 60)

        except Exception as e:
            print(f"[TEMPLATE_LOADER] CRITICAL ERROR: {e}")
            logger.error(f"CRITICAL ERROR in load_all_templates: {e}", exc_info=True)
            raise

    async def load_full_page_template_from_files(
        self, name: str, html_file: Path, css_file: Path, description_file: Path
    ) -> Optional[FullPageTemplate]:
        try:
            print(f"[TEMPLATE_LOADER] Loading files for: {name}")

            html_content = html_file.read_text(encoding='utf-8')
            css_content = css_file.read_text(encoding='utf-8') if css_file.exists() else ""
            description_content = description_file.read_text(encoding='utf-8') if description_file.exists() else ""

            print(f"[TEMPLATE_LOADER] Loaded - HTML: {len(html_content)} chars, CSS: {len(css_content)} chars")

            tags = self._extract_tags_from_html(html_content)

            template = FullPageTemplate(
                name=name,
                html=html_content,
                css=css_content,
                description=description_content,
                tags=tags
            )

            print(f"[TEMPLATE_LOADER] Template object created for: {name}")
            return template

        except Exception as e:
            print(f"[TEMPLATE_LOADER] Error loading template files for {name}: {e}")
            logger.error(f"Error loading template files for {name}: {e}", exc_info=True)
            return None

    async def save_full_page_template_to_qdrant(self, template: FullPageTemplate) -> bool:
        try:
            print(f"[TEMPLATE_LOADER] Generating embedding for: {template.name}")
            embedding = await self._generate_embedding_for_full_page_template(template)
            print(f"[TEMPLATE_LOADER] Generated embedding with {len(embedding)} dimensions")

            result = await self.qdrant.add_template(template, embedding, collection_name=self.collection_name)
            print(f"[TEMPLATE_LOADER] Saved to Qdrant: {result}")
            return result

        except Exception as e:
            print(f"[TEMPLATE_LOADER] Error saving template {template.name}: {e}")
            logger.error(f"Error saving template {template.name} to Qdrant: {e}", exc_info=True)
            raise

    async def search_full_page_templates(
        self, spec: WhitePageSpec, limit: int = 5
    ) -> List[FullPageTemplate]:
        query_text = f"Page Type: {spec.page_type}. Brand: {spec.brand_name}. Business Description: {spec.business_description}. "
        if spec.page_description:
            query_text += f"Desired Page Look: {spec.page_description}. "

        logger.info(f"Searching templates for: {query_text[:100]}...")
        query_embedding = await self._generate_text_embedding(query_text)

        results = await self.qdrant.search_templates(query_embedding, limit, collection_name=self.collection_name)

        logger.info(f"Found {len(results)} templates")

        return [template for template, score in results]

    async def get_full_page_template_by_name(self, name: str) -> Optional[FullPageTemplate]:
        template_data = await self.qdrant.get_template(name, collection_name=self.collection_name)
        if template_data:
            return template_data
        return None

    def _extract_tags_from_html(self, html_content: str) -> List[str]:
        tags = set()
        tag_patterns = {
            "header": ["header", "nav", "navigation"],
            "footer": ["footer", "copyright"],
            "form": ["form", "input", "contact"],
            "button": ["button", "cta", "action"],
            "product": ["product", "item", "catalog"],
            "gallery": ["gallery", "image", "photo"],
            "testimonial": ["testimonial", "review", "feedback"]
        }
        content_lower = html_content.lower()
        for category, keywords in tag_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.add(category)
        return list(tags)

    async def _generate_embedding_for_full_page_template(self, template: FullPageTemplate) -> List[float]:
        text_content = f"{template.name} {template.description} {' '.join(template.tags)}"
        return await self._generate_text_embedding(text_content)

    async def _generate_text_embedding(self, text: str) -> List[float]:
        try:
            embedding = await self.embedding_model.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            return [0.0] * settings.embedding_dim