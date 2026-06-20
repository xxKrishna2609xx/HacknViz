# HacknViz - AI Powered Missing Person Finder

This project implements a web application for face matching and AI-powered image analysis. The backend is built on FastAPI, enabling modularity, type validation, and high performance.

## 📂 Project Structure

The project is split into a decoupled frontend and backend:

* **`Frontend/`**: Contains static assets and templates (`static/`, `templates/index.html`).
* **`Backend/`**: Contains the FastAPI server core:
  * `main.py`: Application entry point and router assembly.
  * `config.py`: Environment validation using Pydantic Settings.
  * `routes/`: Modular endpoints (`match.py`, `analyze.py`, `comprehensive.py`, `search.py`, `health.py`).
  * `services/`: Core logic (`deepface_service.py`, `gemini_service.py`, `cache_service.py`).
  * `utils/`: Common file utilities (`file_utils.py`).

---

## 🚀 Quick Start Guide

### Prerequisites
* Python 3.10+ installed
* A valid Gemini API Key from Google Generative AI

---

### Option A: Unified Server (Recommended)
The FastAPI backend serves **both** the Frontend HTML templates and the Backend API endpoints on a single port (`5000`).

#### 1. Setup Environment
1. Navigate to the `Backend` directory:
   ```bash
   cd Backend
   ```
2. Create and activate a virtual environment:
   * **Windows**: `python -m venv venv` and `venv\Scripts\activate`
   * **Mac/Linux**: `python3 -m venv venv` and `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the `Backend` directory and define your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

#### 2. Start Server
Run the Uvicorn command:
```bash
uvicorn main:app --host 127.0.0.1 --port 5000 --reload
```

#### 3. Access the Project
- **Web Interface**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **API Documentation (Swagger UI)**: [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)
- **Alternative Docs (ReDoc)**: [http://127.0.0.1:5000/redoc](http://127.0.0.1:5000/redoc)

---

### Option B: Split Servers (Frontend & Backend Separate)
If you want to run the Frontend UI and Backend API as separate local services. CORS middleware is fully configured on the backend to allow this.

#### 1. Start Server 1: FastAPI Backend
Open a terminal and start the backend:
```bash
cd Backend
venv\Scripts\activate
uvicorn main:app --host 127.0.0.1 --port 5000 --reload
```

#### 2. Start Server 2: Frontend Static Server
Open a second terminal and start a static web server:
```bash
cd Frontend
python -m http.server 8000
```
The static frontend will be hosted at: [http://127.0.0.1:8000/templates/index.html](http://127.0.0.1:8000/templates/index.html).

#### 3. Update Frontend Fetch Base URLs (If running Split Servers)
If you run the frontend on port `8000`, modify the API `fetch()` endpoints in `Frontend/templates/index.html` from relative paths (e.g. `/search`) to absolute URLs (e.g. `http://127.0.0.1:5000/search`).

---

## 🔒 Note on Image Database & Uploads
User uploads and reference databases are stored locally in the `Backend/uploads` and `Backend/temp_uploads` directories. These directories, along with the `.env` configuration file, are excluded from version control via `.gitignore` to protect privacy and prevent local files from cluttering the Git history.