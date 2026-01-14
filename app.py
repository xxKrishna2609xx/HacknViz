from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import os
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')


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
    
    uploaded_path = os.path.join(UPLOAD_FOLDER, "uploaded.jpg")
    uploaded_file.save(uploaded_path)

    # Get all images from uploads folder to compare against
    # Exclude temporary files used by the application
    excluded_files = {"uploaded.jpg", "analysis_image.jpg", "comprehensive_image.jpg"}
    
    reference_images = [f for f in os.listdir(UPLOAD_FOLDER) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                       and f not in excluded_files]
    
    if not reference_images:
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)
        return jsonify({
            "error": "No reference images found in uploads folder. Please add reference images to the uploads folder.",
            "match_found": False,
            "total_compared": 0
        }), 404

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
            # Use VGG-Face model for face verification
            result = DeepFace.verify(
                uploaded_path,
                cctv_path,
                model_name="VGG-Face",
                enforce_detection=False,
                distance_metric="cosine"
            )
            
            # Get distance and threshold values from DeepFace result
            distance = result.get("distance", float('inf'))
            
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

    # Clean up
    if os.path.exists(uploaded_path):
        os.remove(uploaded_path)

    return jsonify({
        "match_found": any_match_found,
        "total_compared": len(reference_images),
        "best_match": best_match,
        "all_comparisons": comparison_results,
        "threshold": round(match_threshold, 4)
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    uploaded_path = os.path.join(UPLOAD_FOLDER, "analysis_image.jpg")
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
        
        # Clean up
        os.remove(uploaded_path)
        
        return jsonify({
            "analysis": analysis,
            "status": "success"
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/comprehensive', methods=['POST'])
def comprehensive_analysis():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    uploaded_path = os.path.join(UPLOAD_FOLDER, "comprehensive_image.jpg")
    uploaded_file.save(uploaded_path)

    # Get all images from uploads folder to compare against
    excluded_files = {"uploaded.jpg", "analysis_image.jpg", "comprehensive_image.jpg"}
    
    reference_images = [f for f in os.listdir(UPLOAD_FOLDER) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                       and f not in excluded_files]

    results = {
        "face_match": None,
        "ai_analysis": None,
        "status": "success",
        "total_compared": len(reference_images) if reference_images else 0
    }

    # Face matching - compare against all images
    match_threshold = 0.6
    comparison_results = []
    best_match = None
    best_distance = float('inf')
    any_match_found = False
    
    if reference_images:
        for ref_image in reference_images:
            cctv_path = os.path.join(UPLOAD_FOLDER, ref_image)
            
            try:
                result = DeepFace.verify(
                    uploaded_path,
                    cctv_path,
                    model_name="VGG-Face",
                    enforce_detection=False,
                    distance_metric="cosine"
                )
                
                distance = result.get("distance", float('inf'))
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

    # Clean up
    os.remove(uploaded_path)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")