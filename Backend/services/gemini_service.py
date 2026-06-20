import os
import io
import json
import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Any, Optional
from config import settings
from services.cache_service import cache_service

class GeminiService:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def get_image_description(self, filepath: str) -> Optional[str]:
        """
        Generates a concise 2-3 sentence physical description of the person in the image.
        Uses in-memory description cache.
        """
        # Try cache first
        cached_description = cache_service.get_description(filepath)
        if cached_description is not None:
            return cached_description

        # Cache miss - call Gemini
        try:
            image = Image.open(filepath)
            
            prompt = """
            Describe the person in this image in detail, focusing on:
            1. Gender and approximate age group
            2. Clothing: colors, patterns, and types of upper-body clothing (shirts, jackets, hoodies) and lower-body clothing (jeans, pants, shorts, skirts)
            3. Accessories: bags, backpacks, glasses, hats, caps, masks, umbrellas
            4. Hair: color, length, style
            5. Distinctive markings or items
            
            Write a concise 2-3 sentence description of their appearance. Be highly specific about colors and visual markers.
            """
            
            response = self.model.generate_content([prompt, image])
            description = response.text.strip()
            
            if description:
                cache_service.set_description(filepath, description)
                return description
        except Exception as e:
            raise RuntimeError(f"Gemini description generation failed: {str(e)}")

        return None

    def analyze_image(self, filepath: str) -> str:
        """
        Performs standard image analysis for missing person investigations.
        """
        try:
            image = Image.open(filepath)
            
            prompt = """
            Analyze this image and provide detailed information about:
            1. Person description (age, gender, clothing, distinctive features)
            2. Environment/setting details
            3. Any objects or items visible
            4. Potential location clues
            5. Time of day/lighting conditions
            6. Any suspicious or notable activities
            
            Format your response as a structured analysis that could help in a missing person investigation.
            Be detailed but concise, focusing on identifying features and contextual information.
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini image analysis failed: {str(e)}")

    def comprehensive_analysis(self, filepath: str) -> str:
        """
        Detailed investigation analysis for comprehensive route.
        """
        try:
            image = Image.open(filepath)
            
            prompt = """
            Analyze this image for a missing person investigation. Provide:
            1. Detailed person description (age, gender, clothing, distinctive features)
            2. Environment and setting details
            3. Time indicators (lighting, shadows, etc.)
            4. Any objects or items that could help identify location
            5. Behavioral observations
            6. Confidence level in the analysis
            
            Be thorough and professional in your assessment.
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini comprehensive analysis failed: {str(e)}")

    def match_descriptions_to_query(self, query_text: str, records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate how closely each database description matches the query text.
        """
        if not records:
            return []

        prompt = f"""
        You are an AI-powered surveillance assistant. Your task is to match a search query describing a person's appearance against a list of physical descriptions from database images.
        
        Search Query: "{query_text}"
        
        Database Records:
        """
        for i, record in enumerate(records):
            prompt += f"\n{i+1}. Image: {record['image']} | Description: {record['description']}"
        
        prompt += """
        
        Evaluate how closely each record matches the search query. Pay special attention to matching clothes, colors, gender, accessories, and hair.
        
        Provide your evaluation as a raw JSON array of objects. Do not include markdown headers, code blocks (like ```json), or any other conversational text. Return ONLY the valid JSON array matching this format:
        [
          {
            "image": "filename.jpg",
            "score": 85, // Integer percentage score (0 to 100) indicating how well it matches
            "reason": "Explain briefly why this matches or does not match (focus on matching colors, clothing items, etc.)"
          }
        ]
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up potential markdown formatting
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()
                
            results = json.loads(response_text)
            return results
        except Exception as e:
            # Fallback evaluation: return score 0 with error details
            return [{"image": r["image"], "score": 0, "reason": f"Evaluation error: {str(e)}"} for r in records]

# Global gemini service instance
gemini_service = GeminiService()
