from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from src.models.pydantic_models import WhitePageRequest, WhitePageResponse
from src.utils.logging import get_logger
from src.config.settings import settings
from typing import Optional
import os
import asyncio
from pathlib import Path
import re
import aiofiles
import time

from src.tools.template_loader import TemplateLoader
from src.tools.qdrant_manager import AsyncQdrantManager

from src.langgraph_agents.orchestrator_graph import WhitePageOrchestratorGraph

logger = get_logger(__name__)

orchestrator_graph: Optional[WhitePageOrchestratorGraph] = None

GENERATION_DIR = Path("generation")

def ensure_generation_directory() -> None:
    GENERATION_DIR.mkdir(exist_ok=True)
    logger.info(f"Ensured generation directory exists: {GENERATION_DIR.absolute()}")

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'[^\\w\\-_.]', '_', sanitized)
    sanitized = sanitized.strip('._')

    if not sanitized:
        return "page"

    return sanitized[:100]

async def save_generated_page(page_name: str, html_content: str, css_content: str) -> str:
    ensure_generation_directory()

    sanitized_name = sanitize_filename(page_name)
    html_file_path = GENERATION_DIR / f"{sanitized_name}.html"
    css_file_path = GENERATION_DIR / f"{sanitized_name}.css"

    logger.info(f"Saving page '{page_name}' as '{sanitized_name}'")
    logger.info(f"HTML file path: {html_file_path}")
    logger.info(f"CSS file path: {css_file_path}")
    logger.info(f"HTML content length: {len(html_content)}")
    logger.info(f"CSS content length: {len(css_content)}")

    try:
        if css_content.strip():
            async with aiofiles.open(css_file_path, 'w', encoding='utf-8') as css_file:
                await css_file.write(css_content)
            logger.info(f"CSS saved to: {css_file_path}")

            css_link = f'<link rel="stylesheet" href="{sanitized_name}.css">'
            if '<head>' in html_content:
                html_content = html_content.replace('<head>', f'<head>\n    {css_link}')
                logger.info("CSS link added to HTML head")
            else:
                # If no head tag, embed CSS directly (less ideal but functional)
                html_content = f'<style>\n{css_content}\n</style>\n{html_content}'
                logger.info("CSS embedded in HTML (no head tag found)")
        else:
            logger.info("No CSS content to save")

        async with aiofiles.open(html_file_path, 'w', encoding='utf-8') as html_file:
            await html_file.write(html_content)

        if html_file_path.exists():
            file_size = html_file_path.stat().st_size
            logger.info(f"HTML file saved successfully: {html_file_path} (size: {file_size} bytes)")
        else:
            logger.error(f"HTML file was not created: {html_file_path}")
            raise Exception("HTML file was not created")

        return str(html_file_path)

    except Exception as e:
        logger.error(f"Failed to save generated page {page_name}: {e}", exc_info=True)
        raise

async def initialize_orchestrator() -> bool:
    global orchestrator_graph

    try:
        logger.info("=== Starting orchestrator initialization ===")

        # The WhitePageOrchestratorGraph now handles its own Qdrant and LLM initialization
        orchestrator_graph = WhitePageOrchestratorGraph()
        
        # Perform Qdrant health check and template loading via the orchestrator's internal components
        # This assumes the orchestrator's qdrant_manager and template_loader are ready after init
        health_ok = await orchestrator_graph.qdrant_manager.health_check()
        logger.info(f"Qdrant health check result: {health_ok}")
        if not health_ok:
            raise Exception("Qdrant health check failed")

        templates_dir = Path("templates/full_page_html")
        logger.info(f"Loading templates from: {templates_dir.absolute()}")
        if not templates_dir.exists():
            raise Exception(f"Templates directory does not exist: {templates_dir}")
        
        # Load templates into Qdrant via the orchestrator's template loader
        await orchestrator_graph.template_loader.load_all_templates()

        logger.info("=== Orchestrator initialization completed successfully ===")
        return True

    except Exception as e:
        logger.critical(f"Critical error during orchestrator initialization: {e}", exc_info=True)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator_graph

    logger.info("Starting WhitePage Generation Service")
    await initialize_orchestrator()
    yield
    logger.info("Shutting down WhitePage Generation Service")
    if orchestrator_graph:
        await orchestrator_graph.close_connections()

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate", response_model=WhitePageResponse)
async def generate_whitepage(request: WhitePageRequest) -> WhitePageResponse:
    if not orchestrator_graph:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Orchestrator not initialized"
        )

    try:
        start_time = time.time()
        generated_page = await orchestrator_graph.generate_whitepage(request.spec)
        generation_time = time.time() - start_time

        if not generated_page:
            raise Exception("Failed to generate page")

        # Save the generated page
        file_path = await save_generated_page(
            request.spec.page_name,
            generated_page.html,
            generated_page.css
        )

        preview_url = f"/preview/{request.spec.page_name}"

        return WhitePageResponse(
            success=True,
            generated_page=generated_page,
            preview_url=preview_url,
            generation_time=generation_time
        )

    except Exception as e:
        logger.error(f"Failed to generate page: {e}", exc_info=True)
        return WhitePageResponse(
            success=False,
            error=str(e)
        )

@app.get("/preview/{page_name}", response_class=HTMLResponse)
async def preview_page(page_name: str):
    sanitized_name = sanitize_filename(page_name)
    html_file_path = GENERATION_DIR / f"{sanitized_name}.html"

    if not html_file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page '{page_name}' not found"
        )

    async with aiofiles.open(html_file_path, 'r', encoding='utf-8') as file:
        content = await file.read()

    return HTMLResponse(content=content)

@app.get("/generated-pages")
async def list_generated_pages():
    ensure_generation_directory()
    pages = []
    for file in GENERATION_DIR.iterdir():
        if file.is_file() and file.suffix == ".html":
            pages.append(file.name)
    return {"pages": pages}

@app.delete("/generated-pages/{page_name}")
async def delete_generated_page(page_name: str):
    sanitized_name = sanitize_filename(page_name)
    html_file_path = GENERATION_DIR / f"{sanitized_name}.html"
    css_file_path = GENERATION_DIR / f"{sanitized_name}.css"

    try:
        if html_file_path.exists():
            html_file_path.unlink()
        if css_file_path.exists():
            css_file_path.unlink()
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to delete page '{page_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete page: {e}"
        )

@app.get("/health")
async def health_check():
    if not orchestrator_graph:
        return {"status": "unhealthy", "reason": "Orchestrator not initialized"}
    
    qdrant_ok = await orchestrator_graph.qdrant_manager.health_check()
    if not qdrant_ok:
        return {"status": "unhealthy", "reason": "Qdrant connection failed"}
    
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug
    )