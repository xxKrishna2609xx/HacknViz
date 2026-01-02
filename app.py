from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import os
import google.generativeai as genai
import base64
from PIL import Image
import io

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDja0TDdtrgzGqyGxdNxPPDJp7VRq9zdrM"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_faces():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
    uploaded_path = os.path.join(UPLOAD_FOLDER, "uploaded.jpg")
    uploaded_file.save(uploaded_path)

    # Simulated CCTV image
    cctv_path = os.path.join("static", "image.jpg")  # Using existing image.jpg file

    try:
        result = DeepFace.verify(
            uploaded_path,
            cctv_path,
            enforce_detection=False  # skips error if no face is found
        )
        match_found = result["verified"]
    except Exception as e:
        return jsonify({"error": "Not Found", "details": str(e)}), 500

    # Clean up
    os.remove(uploaded_path)

    return jsonify({"match_found": match_found})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
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
    uploaded_path = os.path.join(UPLOAD_FOLDER, "comprehensive_image.jpg")
    uploaded_file.save(uploaded_path)

    # Simulated CCTV image
    cctv_path = os.path.join("static", "image.jpg")

    results = {
        "face_match": None,
        "ai_analysis": None,
        "status": "success"
    }

    # Face matching
    try:
        result = DeepFace.verify(
            uploaded_path,
            cctv_path,
            enforce_detection=False
        )
        results["face_match"] = result["verified"]
    except Exception as e:
        results["face_match"] = False
        results["face_match_error"] = str(e)

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