# Resume Screening Suite

A resume screening application with a React/Vite frontend and a FastAPI backend. Upload PDF, DOCX, or TXT resumes, compare candidates against a job description, and ask questions about the uploaded resumes.

## Project Structure

- `frontend/` React and Vite user interface
- `backend/` FastAPI API and resume processing logic
- `backend/.venv/` Python virtual environment
- `chroma_store/` local vector-store data generated at runtime

## Requirements

- Python 3.12 or compatible Python version
- Node.js and npm
- Google Gemini API key for the configured LLM and embeddings

## Configuration

Create a local `backend/.env` file. Do not commit it. A safe template is available at `backend/.env.example`.

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.3
EMBEDDING_MODEL=gemini-embedding-001
```

Only the Google key is required for the default configuration. Keep API keys private.

## Backend Setup

From the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API from the `backend` directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Health check: `http://localhost:8000/api/health`.

## Frontend Setup

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173` in a browser.

For a production build:

```powershell
npm run build
```

## Workflow

1. Open the frontend and enter or edit the job description.
2. Upload one or more PDF, DOCX, or TXT resumes.
3. Run the analysis to view candidate summaries and detailed analysis.
4. Open Query Results to compare candidates or ask questions about their skills and experience.

## Notes

- The vector store is local and regenerated during analysis.
- The backend includes fallback resume analysis when the external LLM is unavailable.
- Do not commit `.env`, API keys, virtual environments, dependency folders, build output, or generated vector-store data.
