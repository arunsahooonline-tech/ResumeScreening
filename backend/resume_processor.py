import os
import re
import threading
from typing import List, Optional, Sequence

from dotenv import load_dotenv

from google import genai
from google.genai import types

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_classic.retrievers.self_query.base import (
    AttributeInfo,
    SelfQueryRetriever,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY and not GROQ_API_KEY:
    raise RuntimeError(
        "No API key configured. "
        "Set GOOGLE_API_KEY or GROQ_API_KEY in your .env file."
    )


# ============================================================
# CONFIGURATION
# ============================================================

LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "google") or "google").strip().lower()
LLM_MODEL = (os.getenv("LLM_MODEL", "gemini-3.8-flash") or "gemini-3.8-flash").strip()

try:
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
except ValueError:
    LLM_TEMPERATURE = 0.3

EMBEDDING_MODEL = (
    os.getenv("EMBEDDING_MODEL", "gemini-embedding-2") or "gemini-embedding-2"
).strip()

try:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
except ValueError:
    EMBEDDING_DIMENSION = 768

CHROMA_DIRECTORY = (
    os.getenv("CHROMA_DIRECTORY", "chroma_store") or "chroma_store"
).strip()

CHROMA_COLLECTION = (
    os.getenv("CHROMA_COLLECTION", "resume_collection")
    or "resume_collection"
).strip()

DEFAULT_TOP_K = 15

# Serialize Chroma writes/rebuilds inside this FastAPI process.
# This prevents simultaneous /api/analyze requests from modifying the same
# SQLite-backed Chroma collection at the same time.
CHROMA_LOCK = threading.RLock()


# ============================================================
# GEMINI EMBEDDING 2
# ============================================================

class GeminiEmbedding2(Embeddings):
    """
    LangChain-compatible adapter for Google's Gemini Embedding 2 API.

    Important:
    Gemini Embedding 2 can aggregate multiple plain-string inputs into one
    embedding. To guarantee one vector per resume chunk, this adapter embeds
    each document/query separately.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        output_dimensionality: int = 768,
    ):
        if not api_key:
            raise ValueError("Google API key is required.")

        if output_dimensionality < 128 or output_dimensionality > 3072:
            raise ValueError(
                "Gemini Embedding 2 output dimensionality must be between "
                "128 and 3072."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dimensionality = output_dimensionality

    def _embed_one(self, text: str, task_type: str) -> List[float]:
        text = str(text or "").strip()

        if not text:
            # Return a deterministic zero vector only for empty text.
            # Empty resume chunks should normally be filtered before this point.
            return [0.0] * self.output_dimensionality

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
                task_type=task_type,
            ),
        )

        if not response.embeddings:
            raise RuntimeError(
                f"Gemini returned no embedding for model '{self.model}'."
            )

        values = response.embeddings[0].values

        if not values:
            raise RuntimeError("Gemini returned an empty embedding vector.")

        if len(values) != self.output_dimensionality:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected {self.output_dimensionality}, got {len(values)}."
            )

        return list(values)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            self._embed_one(text, "RETRIEVAL_DOCUMENT")
            for text in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text, "RETRIEVAL_QUERY")


# ============================================================
# EMBEDDING INITIALIZATION
# ============================================================

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is required for Gemini Embedding 2."
    )

embeddings_model = GeminiEmbedding2(
    api_key=GOOGLE_API_KEY,
    model=EMBEDDING_MODEL,
    output_dimensionality=EMBEDDING_DIMENSION,
)


# ============================================================
# LLM INITIALIZATION
# ============================================================

if LLM_PROVIDER == "google":
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is required for Google LLM."
        )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

elif LLM_PROVIDER == "groq":
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is required for Groq LLM."
        )

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    )

else:
    raise RuntimeError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
        "Use 'google' or 'groq'."
    )


# ============================================================
# RESUME LOADER
# ============================================================

def load_resume(file_path: str) -> List[Document]:
    """
    Load PDF, DOCX or TXT resume.
    """

    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    }

    extension = os.path.splitext(file_path)[1].lower()
    loader_factory = loaders.get(extension)

    if not loader_factory:
        raise ValueError(
            f"Unsupported file format: {file_path}. "
            "Supported formats: PDF, DOCX, TXT."
        )

    documents = loader_factory(file_path).load()

    if not documents:
        raise ValueError(f"No readable content found in: {file_path}")

    return documents


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def _clean_documents(documents: Sequence[Document]) -> List[Document]:
    """Remove empty pages/chunks while preserving metadata."""
    cleaned = []

    for doc in documents:
        content = (doc.page_content or "").strip()

        if not content:
            continue

        cleaned.append(
            Document(
                page_content=content,
                metadata=dict(doc.metadata or {}),
            )
        )

    return cleaned


def _candidate_name_from_metadata(
    metadata: Optional[dict],
    fallback: str = "Unknown candidate",
) -> str:
    metadata = metadata or {}

    candidate = (
        metadata.get("candidate")
        or metadata.get("source")
        or fallback
    )

    candidate = str(candidate).strip()

    return candidate or fallback


# ============================================================
# RESUME ANALYSIS
# ============================================================

def analyze_resume(
    docs,
    job_description,
    **kwargs,
):
    """
    Analyze one or more resumes against a job description.

    This is the detailed AI analysis used when resumes are uploaded.
    """

    if not job_description or not str(job_description).strip():
        raise ValueError("Job description cannot be empty.")

    docs = _clean_documents(docs)

    if not docs:
        raise ValueError("No resume content was extracted.")

    chunk_size = int(kwargs.get("chunk_size", 1000))
    chunk_overlap = int(kwargs.get("chunk_overlap", 100))

    if chunk_size <= 0:
        chunk_size = 1000

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        chunk_overlap = min(100, chunk_size // 10)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs = text_splitter.split_documents(docs)
    docs = _clean_documents(docs)

    candidate_sections = {}

    for doc in docs:
        candidate = _candidate_name_from_metadata(doc.metadata)
        candidate_sections.setdefault(candidate, []).append(
            doc.page_content
        )

    if not candidate_sections:
        raise ValueError("No candidate content was available for analysis.")

    resume_sections = []

    for candidate, contents in candidate_sections.items():
        candidate_content = "\n".join(contents)

        if len(candidate_content) > 5000:
            candidate_content = candidate_content[:5000] + "..."

        resume_sections.append(
            f"CANDIDATE: {candidate}\n\n{candidate_content}"
        )

    resume_content = "\n\n---\n\n".join(resume_sections)
    candidate_count = len(candidate_sections)

    prompt = f"""
You are a Senior HR Recruiter conducting a professional resume assessment.

Analyze all {candidate_count} candidate resumes against the job requirements.

IMPORTANT:
- Include a separate assessment for EVERY candidate.
- Do not stop after the first candidate.
- Do not combine candidates into one profile.
- Use only evidence present in the supplied resume content.
- Do not invent skills, certifications, employers, dates, or experience.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUMES:
{resume_content}

For EACH candidate provide exactly:

1. CANDIDATE PROFILE
Candidate name, current/most recent position, and years of experience.

2. PROFESSIONAL SUMMARY
Concise assessment of career trajectory, strengths and relevant experience.

3. ALIGNMENT WITH JOB REQUIREMENTS

Matching Skills:
- Relevant matching skills

Missing/Gap Areas:
- Required skills not demonstrated

4. EXPERIENCE ASSESSMENT
Evaluate relevant experience, tenure, progression and industry fit.

5. EDUCATION & CERTIFICATIONS
Relevant education and certifications, if present.

6. SUITABILITY SCORE
NUMBER ONLY, 0-100

7. RECRUITER COMMENTS
Professional and objective assessment.

8. FINAL RECOMMENDATION
YES or NO
"""

    try:
        result = llm.invoke(prompt)

        return (
            result.content
            if hasattr(result, "content")
            else str(result)
        )

    except Exception as exc:
        return _fallback_resume_analysis(
            resume_content,
            job_description,
            str(exc),
        )


# ============================================================
# FALLBACK ANALYSIS
# ============================================================

def _fallback_resume_analysis(
    resume_text: str,
    job_description: str,
    error_message: str = "",
) -> str:
    """
    Conservative fallback used only when the LLM analysis fails.

    It does NOT claim to be equivalent to the AI assessment.
    """

    text = (resume_text or "").lower()
    job = (job_description or "").lower()

    keywords = [
        "python",
        "fastapi",
        "sql",
        "api",
        "aws",
        "docker",
        "ai",
        "machine learning",
        "data",
        "backend",
        "cloud",
        "leadership",
        "engineering",
        "software",
        "javascript",
        "react",
        "angular",
        "dotnet",
        ".net",
        "microservices",
        "postgres",
        "mongodb",
        "kubernetes",
    ]

    matched = [
        keyword
        for keyword in keywords
        if keyword in text and keyword in job
    ]

    score = min(95, max(50, 55 + len(matched) * 6))

    if "experience" in text or "years" in text:
        score += 5

    if "lead" in text or "senior" in text:
        score += 5

    score = min(99, int(score))
    recommendation = "YES" if score >= 70 else "NO"

    return f"""
1. CANDIDATE PROFILE
Fallback assessment based on extracted resume content.

2. PROFESSIONAL SUMMARY
The candidate shows relevant professional experience based on
the extracted resume content.

3. ALIGNMENT WITH JOB REQUIREMENTS

Matching Skills:
- {", ".join(matched[:8]) if matched else "General software engineering"}

Missing/Gap Areas:
- Full AI assessment was unavailable.

4. EXPERIENCE ASSESSMENT
Resume content indicates practical professional experience.

5. EDUCATION & CERTIFICATIONS
Review the original resume.

6. SUITABILITY SCORE
{score}

7. RECRUITER COMMENTS
AI model analysis was unavailable. This result is based on
fallback keyword matching and should be reviewed before making
a hiring decision.

8. FINAL RECOMMENDATION
{recommendation}

Fallback note:
{error_message}
"""


# ============================================================
# CHROMA HELPERS
# ============================================================

def create_vector_store(
    text_chunks,
    persist_directory=CHROMA_DIRECTORY,
    reset=True,
    **kwargs,
):
    """
    Create or update the persistent Chroma vector store.

    IMPORTANT:
    - The Chroma directory and chroma.sqlite3 are NOT deleted on every upload.
    - reset=True removes the existing documents from the collection and then
      inserts the new resume chunks.
    - This keeps the SQLite database persistent and avoids Windows
      PermissionError/WinError 32 caused by deleting an in-use database file.
    """

    text_chunks = _clean_documents(text_chunks)

    if not text_chunks:
        raise ValueError("No resume chunks were supplied.")

    texts = []
    metadatas = []

    for index, chunk in enumerate(text_chunks):
        content = (chunk.page_content or "").strip()

        if not content:
            continue

        candidate = _candidate_name_from_metadata(
            chunk.metadata,
            fallback=f"Candidate {index + 1}",
        )

        texts.append(content)
        metadatas.append(
            {
                "source": candidate,
                "candidate": candidate,
                "chunk_index": index,
            }
        )

    if not texts:
        raise ValueError("No non-empty resume text was available.")

    # Only one request at a time should modify the SQLite-backed collection.
    with CHROMA_LOCK:
        try:
            # Create/open the persistent Chroma collection.
            vector_store = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=embeddings_model,
                persist_directory=persist_directory,
            )

            # --------------------------------------------------------
            # Remove old documents WITHOUT deleting chroma.sqlite3
            # --------------------------------------------------------
            if reset:
                existing = vector_store.get(include=[])

                existing_ids = existing.get("ids", [])

                if existing_ids:
                    vector_store.delete(ids=existing_ids)

                    print(
                        f"Removed {len(existing_ids)} old resume chunks "
                        f"from Chroma collection '{CHROMA_COLLECTION}'."
                    )

            # --------------------------------------------------------
            # Add the new resume chunks
            # --------------------------------------------------------
            vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
            )

            # --------------------------------------------------------
            # Validate the collection
            # --------------------------------------------------------
            count = vector_store._collection.count()

            if count != len(texts):
                raise RuntimeError(
                    "Chroma index validation failed: "
                    f"expected {len(texts)} documents, found {count}."
                )

            print(
                f"Chroma resume index updated successfully: "
                f"{count} chunks, "
                f"{EMBEDDING_MODEL}, "
                f"{EMBEDDING_DIMENSION} dimensions."
            )

            return vector_store

        except Exception as exc:
            print(
                f"Vector store creation/update failed: "
                f"{type(exc).__name__}: {exc}"
            )

            raise RuntimeError(
                f"Could not create/update the resume vector store: {exc}"
            ) from exc

def _open_vector_store(persist_directory: str):
    """Open and validate an existing Chroma resume index."""

    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"Resume index directory does not exist: {persist_directory}"
        )

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )

    count = vector_store._collection.count()

    if count <= 0:
        raise RuntimeError(
            f"Chroma collection '{CHROMA_COLLECTION}' is empty."
        )

    print(
        f"Chroma resume index opened successfully: "
        f"{count} chunks."
    )

    return vector_store


# ============================================================
# SELF QUERY + SEMANTIC SEARCH
# ============================================================

def run_self_query(
    query,
    persist_directory=CHROMA_DIRECTORY,
    **kwargs,
):
    """
    Search the resume vector database and use the LLM to generate
    an HR-oriented answer.

    If SelfQueryRetriever fails, normal semantic similarity search
    is used. If the index itself cannot be opened, a clear error
    is returned instead of pretending that a fallback search occurred.
    """

    query = str(query or "").strip()

    if not query:
        return "Please enter a resume search query."

    try:
        vector_store = _open_vector_store(persist_directory)

    except Exception as exc:
        print(
            f"Vector store open failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            "Resume index is unavailable. "
            "Please upload the resumes again and rebuild the vector store.\n\n"
            f"Technical error: {exc}"
        )

    k = max(1, int(kwargs.get("k", DEFAULT_TOP_K)))

    # --------------------------------------------------------
    # SELF QUERY RETRIEVER
    # --------------------------------------------------------

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source or candidate name of the resume chunk.",
            type="string",
        ),
        AttributeInfo(
            name="candidate",
            description="Candidate name associated with the resume.",
            type="string",
        ),
    ]

    results = []

    try:
        self_query_retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vector_store,
            document_contents=(
                "Resume information containing candidate details, "
                "skills, experience, education and qualifications."
            ),
            metadata_field_info=metadata_field_info,
            search_kwargs={"k": k},
        )

        results = self_query_retriever.invoke(query) or []

    except Exception as exc:
        print(
            f"Self-query retriever failed; using similarity search: "
            f"{type(exc).__name__}: {exc}"
        )

        try:
            results = vector_store.similarity_search(
                query,
                k=k,
            ) or []

        except Exception as search_exc:
            return (
                "Resume semantic search failed.\n\n"
                f"Technical error: {search_exc}"
            )

    # --------------------------------------------------------
    # COLLECT CANDIDATE NAMES
    # --------------------------------------------------------

    candidate_names = []

    try:
        stored_metadata = vector_store.get(
            include=["metadatas"]
        ).get("metadatas", [])

        for metadata in stored_metadata:
            if not metadata:
                continue

            candidate = _candidate_name_from_metadata(
                metadata,
                fallback="",
            )

            if candidate and candidate not in candidate_names:
                candidate_names.append(candidate)

    except Exception as exc:
        print(
            f"Candidate metadata lookup failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # REPRESENTATIVE CHUNKS FOR EACH CANDIDATE
    # --------------------------------------------------------

    all_results = list(results)

    for candidate in candidate_names:
        try:
            candidate_results = vector_store.similarity_search(
                query,
                k=min(3, k),
                filter={"candidate": candidate},
            )

            all_results.extend(candidate_results)

        except Exception as exc:
            print(
                f"Candidate search failed for {candidate}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # REMOVE DUPLICATES AND GROUP BY CANDIDATE
    # --------------------------------------------------------

    source_contents = {}
    seen_content = set()

    result_limit = max(
        k * 2,
        len(candidate_names) * 3,
    )

    for result in all_results[:result_limit]:
        content = (
            result.page_content
            if hasattr(result, "page_content")
            else str(result)
        )

        content = content.strip()

        if not content:
            continue

        metadata = (
            result.metadata
            if hasattr(result, "metadata")
            else {}
        )

        source = _candidate_name_from_metadata(
            metadata,
            fallback="Unknown candidate",
        )

        # Avoid Python's randomized hash for deduplication.
        content_key = content

        if content_key in seen_content:
            continue

        seen_content.add(content_key)

        source_contents.setdefault(source, [])

        if len(source_contents[source]) < 3:
            source_contents[source].append(content)

    if not source_contents:
        return f"No relevant resume documents found for query: {query}"

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    documents_content = []

    for source, contents in source_contents.items():
        for index, content in enumerate(contents, 1):
            if len(contents) > 1:
                documents_content.append(
                    f"[Candidate: {source} - Part {index}]\n{content}"
                )
            else:
                documents_content.append(
                    f"[Candidate: {source}]\n{content}"
                )

    combined_docs = "\n\n---\n\n".join(documents_content)
    sources_list = ", ".join(source_contents.keys())

    # --------------------------------------------------------
    # FINAL HR ANALYSIS
    # --------------------------------------------------------

    analysis_prompt = f"""
You are an HR recruiter analyzing candidate resumes.

USER QUESTION:
{query}

CANDIDATES FOUND:
{sources_list}

RESUME DOCUMENTS:
{combined_docs}

IMPORTANT RULES:
1. Analyze every candidate shown.
2. Clearly identify the candidate for every finding.
3. Do not silently ignore a candidate.
4. Use evidence from the resume.
5. Distinguish demonstrated skills from assumptions.
6. If comparing candidates, describe evidence for each candidate separately.
7. Do not invent qualifications or experience.
8. Keep the answer professional and objective.
9. If the evidence is insufficient, explicitly say so.
10. Do not claim a candidate is the strongest fit unless the supplied
    evidence supports that conclusion.

Provide a clear recruiter-oriented answer.
"""

    try:
        response = llm.invoke(analysis_prompt)

        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

    except Exception as exc:
        print(
            f"LLM analysis error: "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            "The resume search completed, but the final AI analysis "
            "could not be generated.\n\n"
            "Retrieved candidate evidence:\n\n"
            f"{combined_docs}\n\n"
            f"Technical error: {exc}"
        )


# ============================================================
# PARSE ANALYSIS REPORT
# ============================================================

def parse_analysis_report(analysis_text):
    """
    Extract score, recommendation and key reasoning from an
    individual analysis report.
    """

    analysis_text = str(analysis_text or "")

    result = {
        "score": None,
        "recommendation": "Unknown",
        "reasoning": [],
    }

    lines = analysis_text.splitlines()

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    for index, line in enumerate(lines):
        if "SUITABILITY SCORE" in line.upper():
            for next_index in range(
                index + 1,
                min(index + 5, len(lines)),
            ):
                score_text = lines[next_index].strip()

                if not score_text:
                    continue

                match = re.search(r"\b(\d{1,3})\b", score_text)

                if match:
                    score = int(match.group(1))

                    if 0 <= score <= 100:
                        result["score"] = score
                        break

            if result["score"] is not None:
                break

    if result["score"] is None:
        patterns = [
            r"Suitability\s+score[:\s]+(\d{1,3})",
            r"Score[:\s]+(\d{1,3})",
            r"Suitability\s*[:=]\s*(\d{1,3})",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                analysis_text,
                re.IGNORECASE,
            )

            if match:
                score = int(match.group(1))

                if 0 <= score <= 100:
                    result["score"] = score
                    break

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    recommendation_patterns = [
        r"FINAL\s+RECOMMENDATION\s*[:\-]?\s*(YES|NO)\b",
        r"\bRECOMMENDATION\s*[:\-]?\s*(YES|NO)\b",
    ]

    for pattern in recommendation_patterns:
        match = re.search(
            pattern,
            analysis_text,
            re.IGNORECASE,
        )

        if match:
            result["recommendation"] = (
                "Yes" if match.group(1).upper() == "YES" else "No"
            )
            break

    if result["recommendation"] == "Unknown":
        # Look for an exact standalone YES/NO near the end.
        for line in reversed(lines[-10:]):
            value = line.strip().upper().rstrip(".")

            if value == "YES":
                result["recommendation"] = "Yes"
                break

            if value == "NO":
                result["recommendation"] = "No"
                break

    if (
        result["recommendation"] == "Unknown"
        and result["score"] is not None
    ):
        result["recommendation"] = (
            "Yes" if result["score"] >= 70 else "No"
        )

    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    for index, line in enumerate(lines):
        upper_line = line.upper()

        if "KEY" in upper_line and "SKILL" in upper_line:
            for next_index in range(
                index + 1,
                min(index + 8, len(lines)),
            ):
                skill = lines[next_index].strip()

                if (
                    skill
                    and len(skill) > 10
                    and not re.match(r"^\d+\.", skill)
                ):
                    result["reasoning"].append(skill)

                    if len(result["reasoning"]) >= 2:
                        break

            if len(result["reasoning"]) >= 2:
                break

    if not result["reasoning"]:
        bullets = re.findall(
            r"[-•*]\s+(.+?)(?:\n|$)",
            analysis_text,
        )

        result["reasoning"] = [
            bullet.strip()
            for bullet in bullets[:2]
            if len(bullet.strip()) > 15
        ]

    return result
