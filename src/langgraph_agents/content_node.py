from typing import Dict, Any, Optional, List
import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.models.pydantic_models import WhitePageSpec, FullPageTemplate, GeneratedContent
from src.langgraph_agents.state import GraphState, ProcessingStep
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

def clean_llm_json_output(json_string: str) -> str:
    json_string = json_string.strip()
    if json_string.startswith("```json"):
        json_string = json_string[len("```json"):].strip()
    if json_string.endswith("```"):
        json_string = json_string[:-len("```")].strip()
    return json_string

class ContentGenerationNode:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=GeneratedContent)
        self.fallback_content = GeneratedContent(main_content={"error": "Fallback content generation failed"})

    def _get_system_instructions(self, spec: WhitePageSpec, template: FullPageTemplate, previous_content: Optional[GeneratedContent]) -> str:
        products_str = " | ".join(spec.products) if spec.products else "No products specified"
        
        instructions = f"""
        You are a creative content generation agent for HTML templates.
        Create engaging, unique content based on the provided specification while maintaining flexibility in your approach.

        SPECIFICATION GUIDELINES:
        1. Brand name: {spec.brand_name} (use consistently but adapt presentation as needed)
        2. Business context: {spec.business_description} (interpret creatively for content inspiration)
        3. Available products: {products_str} (incorporate naturally into content flow)
        4. Contact details: {spec.contact_email}, {spec.contact_phone}, {spec.address} (include where appropriate)
        5. Page type: {spec.page_type.value if spec.page_type else 'general'} (use as style guidance)
        6. Language preference: adapt to specification data language or enhance for target audience
        7. Content enhancement: feel free to improve existing text, comments, or descriptions for better engagement
        8. User-generated content: create authentic-feeling reviews, testimonials, or comments when present

        CONTENT APPROACH:
        - Prioritize uniqueness and originality over strict template adherence
        - Create compelling, realistic product/service descriptions with market-appropriate pricing
        - Develop content that resonates with the target business audience
        - Balance specification requirements with creative content development
        - Adapt tone and style to match business type and customer expectations

        OUTPUT STRUCTURE:
        Return JSON with these fields (adapt content to fit naturally):
        - main_content: object with title, description, hero_text fields
        - items: array of objects with name, description, price fields
        - images: object with image descriptions
        - contact_info: object with email, phone, address fields
        - other_data: object with button_text, meta_description fields

        Use the GeneratedContent model structure for your response.
        """
        
        if previous_content:
            instructions += f"\n\nPrevious content for reference and improvement: {previous_content.model_dump_json()}"

        return instructions

    def _correct_llm_data_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        corrected = data.copy()

        if "main_content" in corrected:
            if isinstance(corrected["main_content"], str):
                logger.warning("Converting main_content from string to dict")
                corrected["main_content"] = {"content": corrected["main_content"]}
            elif not isinstance(corrected["main_content"], dict):
                logger.warning(f"Invalid main_content type: {type(corrected['main_content'])}, setting to None")
                corrected["main_content"] = None

        if "images" in corrected:
            if isinstance(corrected["images"], list):
                logger.warning("Converting images from list to dict")
                images_dict = {}
                for i, img in enumerate(corrected["images"]):
                    images_dict[f"image_{i+1}"] = self._extract_url_value(img)
                corrected["images"] = images_dict
            elif isinstance(corrected["images"], dict):
                logger.debug("Processing images dict structure")
                images_dict = {}
                for key, value in corrected["images"].items():
                    images_dict[key] = self._extract_url_value(value)
                corrected["images"] = images_dict
            elif not isinstance(corrected["images"], dict):
                logger.warning(f"Invalid images type: {type(corrected['images'])}, setting to None")
                corrected["images"] = None

        if "items" in corrected:
            if isinstance(corrected["items"], dict):
                logger.warning("Converting items from dict to list")
                corrected["items"] = [corrected["items"]]
            elif isinstance(corrected["items"], list):
                fixed_items = []
                for item in corrected["items"]:
                    if isinstance(item, dict):
                        fixed_items.append(item)
                    else:
                        logger.warning(f"Converting item {item} to dict")
                        fixed_items.append({"name": str(item)})
                corrected["items"] = fixed_items

        if "contact_info" in corrected:
            if not isinstance(corrected["contact_info"], dict):
                logger.warning(f"Invalid contact_info type: {type(corrected['contact_info'])}, setting to None")
                corrected["contact_info"] = None

        if "other_data" in corrected:
            if not isinstance(corrected["other_data"], dict):
                logger.warning(f"Invalid other_data type: {type(corrected['other_data'])}, setting to None")
                corrected["other_data"] = None

        return corrected

    def _extract_url_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            if "url" in value:
                return str(value["url"])
            elif "src" in value:
                return str(value["src"])
            elif "href" in value:
                return str(value["href"])
            else:
                return str(list(value.values())[0]) if value else ""
        else:
            return str(value)

    async def content_generation_node(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Executing content generation node for {state['brand_name']}")

        selected_template = state.get("selected_template")
        if not selected_template:
            error_msg = "No template selected for content generation."
            logger.error(error_msg)
            return {
                **state,
                "generated_content": self.fallback_content,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "content_generation", "message": error_msg}
                ]
            }

        try:
            spec = state["spec"]
            
            system_message = self._get_system_instructions(
                spec, selected_template, state.get("generated_content")
            )
            
            human_message = f"""
            Generate content for:
            Brand: {spec.brand_name}
            Page Type: {spec.page_type.value if spec.page_type else 'general'}
            Business: {spec.business_description}
            Template: {selected_template.description}

            Contact Details:
            Email: {spec.contact_email}
            Phone: {spec.contact_phone}
            Address: {spec.address}

            Products: {json.dumps(spec.products) if spec.products else 'No specific products'}

            Generate structured content following the GeneratedContent model.
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", human_message)
            ])
            
            chain = prompt | self.llm | self.parser
            
            llm_response = await chain.ainvoke({})
            
            if isinstance(llm_response, GeneratedContent):
                corrected_data = self._correct_llm_data_types(llm_response.model_dump())
            else:
                corrected_data = self._correct_llm_data_types(llm_response)
            
            generated_content = GeneratedContent(**corrected_data)

            return {
                **state,
                "generated_content": generated_content,
                "messages": state.get("messages", []) + [
                    {"type": "info", "step": "content_generation", "message": "Content generated successfully"}
                ]
            }

        except Exception as e:
            error_msg = f"Error in content generation node: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                **state,
                "generated_content": self.fallback_content,
                "messages": state.get("messages", []) + [
                    {"type": "error", "step": "content_generation", "message": error_msg}
                ]
            }