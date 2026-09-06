import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .resume_processor import analyze_resume, create_vector_store, load_resume, parse_analysis_report, run_self_query
except ImportError:
    from resume_processor import analyze_resume, create_vector_store, load_resume, parse_analysis_report, run_self_query

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "chroma_store"

app = FastAPI(
    title="Resume Screening API",
    version="1.0.0",
    description="FastAPI service for resume parsing, AI analysis, and candidate screening.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "resume-screening-api"}


class QueryRequest(BaseModel):
    question: str


@app.post("/api/analyze")
async def analyze_resumes(
    job_description: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one resume file is required.")

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is required.")

    uploaded_paths: list[str] = []
    documents: list = []

    try:
        for upload_index, upload in enumerate(files, 1):
            if upload.filename is None or not upload.filename.strip():
                continue

            suffix = Path(upload.filename).suffix.lower()
            if suffix not in {".pdf", ".docx", ".txt"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {upload.filename}. Only PDF, DOCX, and TXT are allowed.",
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await upload.read()
                temp_file.write(content)
                uploaded_paths.append(temp_file.name)

            resume_docs = load_resume(temp_file.name)
            candidate_name = f"Candidate {upload_index}: {Path(upload.filename).name}"
            for document in resume_docs:
                document.metadata["candidate"] = candidate_name
            documents.extend(resume_docs)

        if not documents:
            raise HTTPException(status_code=400, detail="No valid resume content could be extracted.")

        create_vector_store(documents, persist_directory=str(PERSIST_DIR))
        try:
            analysis = analyze_resume(documents, job_description)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"AI analysis failed: {exc}") from exc

        parsed = parse_analysis_report(analysis)

        return {
            "success": True,
            "analysis": analysis,
            "parsed": parsed,
            "document_count": len(documents),
        }
    finally:
        for path in uploaded_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post("/api/query")
async def resume_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = run_self_query(request.question, persist_directory=str(PERSIST_DIR))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Query failed: {exc}") from exc

    return {"success": True, "answer": answer}


@app.get("/")
def root():
    return {"message": "Resume Screening API is running."}
