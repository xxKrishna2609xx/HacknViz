import os
import numpy as np
from deepface import DeepFace
from typing import List, Dict, Any, Optional, Tuple
from config import settings
from services.cache_service import cache_service

class DeepFaceService:
    def __init__(self, model_name: str = "VGG-Face", enforce_detection: bool = False):
        self.model_name = model_name
        self.enforce_detection = enforce_detection

    def get_embedding(self, filepath: str) -> Optional[List[float]]:
        """
        Get the face embedding for an image. Checks the cache first.
        If cache is missing or invalid, calculates it using DeepFace.
        """
        # Try cache first
        embedding = cache_service.get_embedding(filepath)
        if embedding is not None:
            return embedding
        
        # Cache miss - calculate using DeepFace
        try:
            reprs = DeepFace.represent(
                img_path=filepath, 
                model_name=self.model_name, 
                enforce_detection=self.enforce_detection
            )
            if reprs and len(reprs) > 0:
                embedding = reprs[0]["embedding"]
                cache_service.set_embedding(filepath, embedding)
                return embedding
        except Exception as e:
            # Re-raise or log for route layer handling
            raise RuntimeError(f"DeepFace embedding generation failed: {str(e)}")
        
        return None

    def calculate_distance(self, embedding_a: List[float], embedding_b: List[float]) -> float:
        """
        Calculate cosine distance between two embeddings.
        D = 1 - (A . B) / (||A|| * ||B||)
        """
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
            
        similarity = dot_product / (norm_a * norm_b)
        return float(1.0 - similarity)

    def match_face(self, query_filepath: str) -> Dict[str, Any]:
        """
        Compare the query image against all reference images in the uploads folder.
        """
        # Check reference images in uploads directory
        uploads_dir = str(settings.UPLOAD_FOLDER)
        if not os.path.exists(uploads_dir):
            raise FileNotFoundError("Reference uploads directory does not exist.")

        reference_images = [f for f in os.listdir(uploads_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not reference_images:
            return {
                "match_found": False,
                "total_compared": 0,
                "best_match": None,
                "all_comparisons": [],
                "threshold": settings.MATCH_THRESHOLD
            }

        # Calculate query embedding
        query_embedding = self.get_embedding(query_filepath)
        if not query_embedding:
            raise ValueError("Could not extract face features from query image.")

        comparison_results = []
        best_match = None
        best_distance = float('inf')
        any_match_found = False
        
        for ref_image in reference_images:
            ref_path = os.path.join(uploads_dir, ref_image)
            
            try:
                ref_embedding = self.get_embedding(ref_path)
                if ref_embedding is None:
                    raise ValueError("Could not extract face embedding from reference image.")
                
                distance = self.calculate_distance(query_embedding, ref_embedding)
                match_found = distance < settings.MATCH_THRESHOLD
                
                # Confidence calculation
                if distance <= 0:
                    confidence = 100.0
                elif distance >= 1:
                    confidence = 0.0
                else:
                    confidence = (1 - distance) * 100
                
                result = {
                    "image": ref_image,
                    "match_found": match_found,
                    "distance": round(distance, 4),
                    "confidence": round(confidence, 2),
                    "threshold": round(settings.MATCH_THRESHOLD, 4)
                }
                
                comparison_results.append(result)
                
                # Track best match
                if match_found and distance < best_distance:
                    best_match = result
                    best_distance = distance
                    any_match_found = True
                elif not any_match_found and distance < best_distance:
                    best_match = result
                    best_distance = distance
                    
            except Exception as e:
                # Append error result but keep matching other images
                comparison_results.append({
                    "image": ref_image,
                    "match_found": False,
                    "error": str(e),
                    "distance": None,
                    "confidence": 0.0
                })

        # Sort results by distance (ascending) so best match is first
        comparison_results.sort(key=lambda x: x.get("distance") if x.get("distance") is not None else float('inf'))

        return {
            "match_found": any_match_found,
            "total_compared": len(reference_images),
            "best_match": best_match,
            "all_comparisons": comparison_results,
            "threshold": round(settings.MATCH_THRESHOLD, 4)
        }

# Global deepface service instance
deepface_service = DeepFaceService()
