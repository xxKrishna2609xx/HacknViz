from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import os
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='../Frontend/templates', static_folder='../Frontend/static')
UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Configure Gemini API from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# In-memory cache for reference image embeddings to avoid recalculating them on every request
# Key: absolute file path -> Value: (file_modification_time, embedding_vector)
REFERENCE_EMBEDDINGS = {}

def get_image_embedding(path):
    global REFERENCE_EMBEDDINGS
    try:
        mtime = os.path.getmtime(path)
        cached = REFERENCE_EMBEDDINGS.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        
        # Calculate representation using DeepFace
        reprs = DeepFace.represent(img_path=path, model_name="VGG-Face", enforce_detection=False)
        if reprs and len(reprs) > 0:
            embedding = reprs[0]["embedding"]
            REFERENCE_EMBEDDINGS[path] = (mtime, embedding)
            return embedding
    except Exception as e:
        print(f"Error generating embedding for {path}: {e}")
    return None

# In-memory cache for reference image descriptions
# Key: absolute file path -> Value: (file_modification_time, description_text)
REFERENCE_DESCRIPTIONS = {}

def get_image_description(path):
    global REFERENCE_DESCRIPTIONS
    try:
        mtime = os.path.getmtime(path)
        cached = REFERENCE_DESCRIPTIONS.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        
        # Open and describe reference image
        with open(path, 'rb') as image_file:
            image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data))
        
        prompt = """
        Describe the person in this image in detail, focusing on:
        1. Gender and approximate age group
        2. Clothing: colors, patterns, and types of upper-body clothing (shirts, jackets, hoodies) and lower-body clothing (jeans, pants, shorts, skirts)
        3. Accessories: bags, backpacks, glasses, hats, caps, masks, umbrellas
        4. Hair: color, length, style
        5. Distinctive markings or items
        
        Write a concise 2-3 sentence description of their appearance. Be highly specific about colors and visual markers.
        """
        
        response = model.generate_content([prompt, image])
        description = response.text.strip()
        
        if description:
            REFERENCE_DESCRIPTIONS[path] = (mtime, description)
            return description
    except Exception as e:
        print(f"Error generating description for {path}: {e}")
    return None

import json

def match_descriptions_to_query(query_text, records):
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
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response text if Gemini wraps it in code blocks
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
        print(f"Error matching descriptions to query: {e}")
        # Fallback: return score 0 for all
        return [{"image": r["image"], "score": 0, "reason": f"Matcher error: {str(e)}"} for r in records]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_faces():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "File type not allowed. Supported formats are png, jpg, jpeg."}), 400
    
    ext = uploaded_file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{ext}"
    uploaded_path = os.path.join(TEMP_FOLDER, unique_filename)
    uploaded_file.save(uploaded_path)

    try:
        # Get all images from uploads folder to compare against
        reference_images = [f for f in os.listdir(UPLOAD_FOLDER) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not reference_images:
            return jsonify({
                "error": "No reference images found in uploads folder. Please add reference images to the uploads folder.",
                "match_found": False,
                "total_compared": 0
            }), 404

        # Get query embedding
        query_reprs = DeepFace.represent(img_path=uploaded_path, model_name="VGG-Face", enforce_detection=False)
        if not query_reprs or len(query_reprs) == 0:
            return jsonify({"error": "Failed to extract features from the uploaded image."}), 500
        
        query_embedding = query_reprs[0]["embedding"]

        # Match threshold
        match_threshold = 0.6
        model_threshold = 0.68  # Default threshold for VGG-Face with cosine
        
        # Compare against all images
        comparison_results = []
        best_match = None
        best_distance = float('inf')
        any_match_found = False
        
        for ref_image in reference_images:
            cctv_path = os.path.join(UPLOAD_FOLDER, ref_image)
            
            try:
                ref_embedding = get_image_embedding(cctv_path)
                if ref_embedding is None:
                    raise ValueError("Could not extract face embedding from reference image.")
                
                # Calculate cosine distance using numpy
                import numpy as np
                a = np.array(query_embedding)
                b = np.array(ref_embedding)
                distance = float(1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
                
                # Determine if faces match based on our threshold
                match_found = distance < match_threshold
                
                # Calculate confidence percentage
                if distance <= 0:
                    confidence = 100.0
                elif distance >= 1:
                    confidence = 0.0
                else:
                    confidence = (1 - distance) * 100
                
                comparison_result = {
                    "image": ref_image,
                    "match_found": match_found,
                    "distance": round(distance, 4),
                    "confidence": round(confidence, 2),
                    "threshold": round(match_threshold, 4)
                }
                
                comparison_results.append(comparison_result)
                
                # Track best match
                if match_found and distance < best_distance:
                    best_match = comparison_result
                    best_distance = distance
                    any_match_found = True
                elif not any_match_found and distance < best_distance:
                    best_match = comparison_result
                    best_distance = distance
                    
            except Exception as e:
                # If comparison fails for one image, continue with others
                comparison_results.append({
                    "image": ref_image,
                    "match_found": False,
                    "error": str(e),
                    "distance": None,
                    "confidence": 0.0
                })
        
        # Sort results by distance (best matches first)
        comparison_results.sort(key=lambda x: x.get("distance", float('inf')))

        return jsonify({
            "match_found": any_match_found,
            "total_compared": len(reference_images),
            "best_match": best_match,
            "all_comparisons": comparison_results,
            "threshold": round(match_threshold, 4)
        })
    finally:
        # Clean up
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "File type not allowed. Supported formats are png, jpg, jpeg."}), 400
        
    ext = uploaded_file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{ext}"
    uploaded_path = os.path.join(TEMP_FOLDER, unique_filename)
    uploaded_file.save(uploaded_path)

    try:
        # Read and process the image
        with open(uploaded_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(image_data))
        
        # Analyze with Gemini
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
        
        response = model.generate_content([prompt, image])
        analysis = response.text
        
        return jsonify({
            "analysis": analysis,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        # Clean up
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)

@app.route('/comprehensive', methods=['POST'])
def comprehensive_analysis():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "File type not allowed. Supported formats are png, jpg, jpeg."}), 400
        
    ext = uploaded_file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{ext}"
    uploaded_path = os.path.join(TEMP_FOLDER, unique_filename)
    uploaded_file.save(uploaded_path)

    try:
        # Get all images from uploads folder to compare against
        reference_images = [f for f in os.listdir(UPLOAD_FOLDER) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        results = {
            "face_match": None,
            "ai_analysis": None,
            "status": "success",
            "total_compared": len(reference_images) if reference_images else 0
        }

        # Get query embedding
        query_reprs = DeepFace.represent(img_path=uploaded_path, model_name="VGG-Face", enforce_detection=False)
        if query_reprs and len(query_reprs) > 0:
            query_embedding = query_reprs[0]["embedding"]
        else:
            query_embedding = None

        # Face matching - compare against all images
        match_threshold = 0.6
        comparison_results = []
        best_match = None
        best_distance = float('inf')
        any_match_found = False
        
        if reference_images and query_embedding is not None:
            for ref_image in reference_images:
                cctv_path = os.path.join(UPLOAD_FOLDER, ref_image)
                
                try:
                    ref_embedding = get_image_embedding(cctv_path)
                    if ref_embedding is None:
                        raise ValueError("Could not extract face embedding from reference image.")
                    
                    # Calculate cosine distance using numpy
                    import numpy as np
                    a = np.array(query_embedding)
                    b = np.array(ref_embedding)
                    distance = float(1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
                    
                    match_found = distance < match_threshold
                    
                    if distance <= 0:
                        confidence = 100.0
                    elif distance >= 1:
                        confidence = 0.0
                    else:
                        confidence = (1 - distance) * 100
                    
                    comparison_result = {
                        "image": ref_image,
                        "match_found": match_found,
                        "distance": round(distance, 4),
                        "confidence": round(confidence, 2)
                    }
                    
                    comparison_results.append(comparison_result)
                    
                    if match_found and distance < best_distance:
                        best_match = comparison_result
                        best_distance = distance
                        any_match_found = True
                    elif not any_match_found and distance < best_distance:
                        best_match = comparison_result
                        best_distance = distance
                        
                except Exception as e:
                    comparison_results.append({
                        "image": ref_image,
                        "match_found": False,
                        "error": str(e),
                        "distance": None,
                        "confidence": 0.0
                    })
            
            # Sort results by distance
            comparison_results.sort(key=lambda x: x.get("distance", float('inf')))
            results["face_match"] = any_match_found
            results["best_match"] = best_match
            results["all_comparisons"] = comparison_results
            results["face_match_threshold"] = round(match_threshold, 4)
        else:
            results["face_match"] = False
            if query_embedding is None:
                results["face_match_error"] = "Failed to extract face features from the query image."
            else:
                results["face_match_error"] = "No reference images found in uploads folder"

        # AI Analysis
        try:
            with open(uploaded_path, 'rb') as image_file:
                image_data = image_file.read()
            
            image = Image.open(io.BytesIO(image_data))
            
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
            
            response = model.generate_content([prompt, image])
            results["ai_analysis"] = response.text
            
        except Exception as e:
            results["ai_analysis"] = f"AI analysis failed: {str(e)}"

        return jsonify(results)
    finally:
        # Clean up
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)

@app.route('/search', methods=['POST'])
def search_appearance():
    query = request.form.get('query')
    if not query and request.is_json:
        query = request.json.get('query')
        
    uploaded_file = request.files.get('image')

    # If it's an image search, we first describe the uploaded image
    if uploaded_file and uploaded_file.filename != '':
        if not allowed_file(uploaded_file.filename):
            return jsonify({"error": "File type not allowed. Supported formats are png, jpg, jpeg."}), 400
            
        ext = uploaded_file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"query_{uuid.uuid4()}.{ext}"
        uploaded_path = os.path.join(TEMP_FOLDER, unique_filename)
        uploaded_file.save(uploaded_path)
        
        try:
            # Describe the query image
            with open(uploaded_path, 'rb') as image_file:
                image_data = image_file.read()
            image = Image.open(io.BytesIO(image_data))
            
            prompt = """
            Describe this person's appearance in detail for a missing person search. Focus on:
            1. Gender and approximate age group
            2. Clothing colors and types (both upper and lower body)
            3. Accessories (bags, hats, glasses, backpacks)
            4. Hair color and style
            
            Provide a highly descriptive 2-3 sentence summary.
            """
            
            response = model.generate_content([prompt, image])
            query = response.text.strip()
            
            if not query:
                return jsonify({"error": "Failed to analyze the uploaded search image."}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to process query image: {str(e)}"}), 500
        finally:
            if os.path.exists(uploaded_path):
                os.remove(uploaded_path)
    
    # If we have no query text and no image description, fail
    if not query:
        return jsonify({"error": "No search query or image provided."}), 400

    # Get reference images and their descriptions
    try:
        reference_images = [f for f in os.listdir(UPLOAD_FOLDER) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not reference_images:
            return jsonify({"error": "No reference images in the uploads database."}), 404
            
        records = []
        for ref_image in reference_images:
            path = os.path.join(UPLOAD_FOLDER, ref_image)
            description = get_image_description(path)
            if description:
                records.append({
                    "image": ref_image,
                    "description": description
                })
        
        # Batch match descriptions to query
        matches = match_descriptions_to_query(query, records)
        
        # Sort matches by score descending
        for m in matches:
            try:
                m["score"] = int(m.get("score", 0))
            except:
                m["score"] = 0
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return jsonify({
            "status": "success",
            "search_query": query,
            "results": matches
        })
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")