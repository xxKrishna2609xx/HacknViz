from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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
    cctv_path = os.path.join("static", "cctv_image.jpg")  # Ensure this file exists

    try:
        result = DeepFace.verify(
            uploaded_path,
            cctv_path,
            enforce_detection=False  # skips error if no face is found
        )
        match_found = result["verified"]
    except Exception as e:
        return jsonify({"error": f"Face comparison failed: {str(e)}"}), 500

    # Clean up
    os.remove(uploaded_path)

    return jsonify({"match_found": match_found})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")