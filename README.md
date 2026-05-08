# HacknViz

This project implements a web application for face matching and AI-powered image analysis.

## Project Structure

The project has been organized into a separated frontend and backend structure:

* **`Frontend/`**: Contains all static assets and HTML templates (CSS, JS, images, `index.html`).
* **`Backend/`**: Contains the core Flask server (`app.py`), the virtual environment (`venv`), and handles file uploads and integrations (DeepFace, Gemini AI).

## Quick Start Guide

### Prerequisites
* Python 3.8+ installed
* A valid Gemini API Key from Google Generative AI

### 1. Setup the Backend Environment
1. Open your terminal and navigate to the Backend directory:
   ```bash
   cd Backend
   ```
2. Create and activate a virtual environment (if you haven't already):
   * **Windows**: `python -m venv venv` and then `venv\Scripts\activate`
   * **Mac/Linux**: `python3 -m venv venv` and then `source venv/bin/activate`

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Configure Environment Variables
1. Ensure there is a `.env` file in the `Backend` folder.
2. Add your Gemini API key to the `.env` file:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

### 3. Run the Application
1. Start the Flask development server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

### Note on Image Uploads
User uploads are stored in the `Backend/uploads` directory. This directory is intentionally omitted from version control (via `.gitignore`) to protect user privacy and avoid cluttering the repository.