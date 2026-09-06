import os
import re
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.self_query.base import AttributeInfo, SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
if google_key:
    os.environ["GOOGLE_API_KEY"] = google_key
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

llm_provider = (os.getenv("LLM_PROVIDER", "google") or "google").strip().lower()
llm_model_name = (os.getenv("LLM_MODEL", "") or "").strip()
llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
embedding_model_name = (os.getenv("EMBEDDING_MODEL", "gemini-embedding-001") or "gemini-embedding-001").strip()

valid_google_model = "gemini-2.5-flash"

if llm_provider == "google" and google_key:
    llm_provider = "google"
    if not llm_model_name or llm_model_name in {"gemini-2.0-flash", "gemini-2.0-flash-latest", "gemini-1.5-flash", "gemini-2.5-flash-lite", "llama-3.3-80b-versatile", "llama-3.3-70b-versatile"}:
        llm_model_name = valid_google_model
    llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=llm_temperature, google_api_key=google_key)
elif llm_provider == "groq" and groq_key:
    if not llm_model_name or llm_model_name in {"llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama-3.3-80b-versatile", "llama-3.3-70b-versatile"}:
        llm_model_name = "llama-3.3-70b-versatile"
    llm = ChatGroq(model=llm_model_name, temperature=llm_temperature, api_key=groq_key)
elif google_key:
    llm_provider = "google"
    if not llm_model_name or llm_model_name in {"gemini-2.0-flash", "gemini-2.0-flash-latest", "gemini-1.5-flash", "gemini-2.5-flash-lite", "llama-3.3-80b-versatile", "llama-3.3-70b-versatile"}:
        llm_model_name = valid_google_model
    llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=llm_temperature, google_api_key=google_key)
else:
    raise RuntimeError("No supported LLM API key is configured. Set GOOGLE_API_KEY or GROQ_API_KEY.")

embeddings_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=google_key)


def _fallback_resume_analysis(resume_text: str, job_description: str, error_message: str = "") -> str:
    text = resume_text.strip()
    lower_text = text.lower()
    job_lower = job_description.lower()

    keywords = [
        "python", "fastapi", "sql", "api", "aws", "docker", "ai", "machine learning",
        "data", "backend", "cloud", "leadership", "engineering", "software", "javascript",
        "react", "microservices", "postgres", "mongodb", "kubernetes"
    ]
    matched = [word for word in keywords if word in lower_text and word in job_lower]
    score = min(95, max(50, 55 + len(matched) * 6))
    if "experience" in lower_text or "years" in lower_text:
        score += 5
    if "lead" in lower_text or "senior" in lower_text:
        score += 5

    score = min(99, int(score))
    recommendation = "YES" if score >= 70 else "NO"

    return (
        "1. CANDIDATE PROFILE\n"
        "Fallback resume assessment based on extracted profile content.\n\n"
        "2. PROFESSIONAL SUMMARY\n"
        "The candidate shows measurable experience in software engineering and relevant technology work. The summary is based on the uploaded resume text and fallback screening rules because the AI model endpoint is unavailable.\n\n"
        "3. ALIGNMENT WITH JOB REQUIREMENTS\n"
        f"Matching Skills:\n- Relevant keywords detected: {', '.join(matched[:6]) if matched else 'general software engineering experience'}\n\nMissing/Gap Areas:\n- Some role-specific details may be missing from the resume text if the document is incomplete or abbreviated.\n\n"
        "4. EXPERIENCE ASSESSMENT\n"
        "The candidate profile indicates practical engineering experience and relevant tools, but a full scoring pass requires a valid AI model connection.\n\n"
        "5. EDUCATION & CERTIFICATIONS\n"
        "Education and certifications were not fully parsed from the fallback mode; please review the original resume file manually if needed.\n\n"
        "6. SUITABILITY SCORE\n"
        f"{score}\n\n"
        "7. RECRUITER COMMENTS\n"
        "This candidate shows promising technical experience for an engineering role based on the resume text and keyword fit, but final assessment should be confirmed with a live AI model or recruiter review.\n\n"
        "8. FINAL RECOMMENDATION\n"
        f"{recommendation}\n\n"
        f"Fallback note: {error_message}\n"
    )


def load_resume(file_path):
    loaders = {
        ".pdf": lambda fp: PyPDFLoader(fp),
        ".docx": lambda fp: Docx2txtLoader(fp),
        ".txt": lambda fp: TextLoader(fp, encoding="utf8"),
    }

    ext = os.path.splitext(file_path)[1].lower()
    loader = loaders.get(ext, lambda fp: None)(file_path)

    if loader is None:
        raise ValueError(f"Unsupported file format: {file_path}. Supported formats: pdf, docx, txt")

    return loader.load()


def analyze_resume(docs, job_description, **kwargs):
    chunk_size = kwargs.get("chunk_size", 1000)
    chunk_overlap = kwargs.get("chunk_overlap", 100)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(docs)

    candidate_sections = {}
    for doc in docs:
        candidate = doc.metadata.get("candidate") or doc.metadata.get("source") or "Unknown candidate"
        candidate_sections.setdefault(candidate, []).append(doc.page_content)

    resume_sections = []
    for candidate, contents in candidate_sections.items():
        candidate_content = "\n".join(contents)
        if len(candidate_content) > 3000:
            candidate_content = candidate_content[:3000] + "..."
        resume_sections.append(f"CANDIDATE: {candidate}\n{candidate_content}")

    resume_content = "\n\n---\n\n".join(resume_sections)
    candidate_count = len(candidate_sections)

    prompt = f"""You are a Senior HR Recruiter conducting a professional resume assessment. Analyze all {candidate_count} candidate resumes against the job requirements and provide a comprehensive evaluation.

IMPORTANT: Include a separate assessment for EVERY candidate shown below. Do not stop after the first candidate or combine candidates into one profile.

PROVIDE RESPONSE IN THIS EXACT FORMAT:

For each candidate, use the following sections and clearly include the candidate name:

1. CANDIDATE PROFILE
[Candidate name, current/most recent position, years of experience summary - 1-2 sentences]

2. PROFESSIONAL SUMMARY
[Concise assessment of the candidate's overall career trajectory and key strengths - 2-3 sentences]

3. ALIGNMENT WITH JOB REQUIREMENTS
Matching Skills:
[Bullet points of relevant skills that match job requirements]

Missing/Gap Areas:
[Bullet points of required skills not demonstrated in resume]

4. EXPERIENCE ASSESSMENT
[Evaluation of relevant experience, tenure, progression, and industry fit - 2-3 sentences]

5. EDUCATION & CERTIFICATIONS
[Relevant qualifications and certifications]

6. SUITABILITY SCORE
[NUMBER ONLY, 0-100]

7. RECRUITER COMMENTS
[Professional, objective assessment of fit as a senior HR recruiter would note - 2-3 sentences]

8. FINAL RECOMMENDATION
YES or NO

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUMES:
{resume_content}
"""

    try:
        result = llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)
    except Exception as exc:
        return _fallback_resume_analysis(resume_content, job_description, str(exc))


def create_vector_store(text_chunks, persist_directory="chroma_store", **kwargs):
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except PermissionError:
            print(f"Warning: Could not delete {persist_directory} (locked by another process).")
            print("Proceeding with existing store. If dimension mismatch errors occur, restart the app.")

    texts = [chunk.page_content for chunk in text_chunks]
    if not texts:
        return None

    try:
        metadatas = []
        for i, chunk in enumerate(text_chunks):
            candidate = chunk.metadata.get("candidate") or chunk.metadata.get("source") or f"Candidate {i + 1}"
            metadatas.append({"source": candidate, "candidate": candidate, "chunk_index": i})
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=embeddings_model,
            metadatas=metadatas,
            persist_directory=persist_directory,
            **kwargs,
        )
        return vector_store
    except Exception as exc:
        print(f"Vector store creation failed; using fallback mode. Error: {exc}")
        return None


def run_self_query(query, persist_directory="chroma_store", **kwargs):
    try:
        vector_store = Chroma(
            embedding_function=embeddings_model,
            persist_directory=persist_directory,
        )
    except Exception as exc:
        print(f"Vector-store open failed; using fallback query answer. Error: {exc}")
        return f"I could not open the saved resume index. A fallback search is being used. Query: {query}"

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source of the resume chunk",
            type="string",
        )
    ]

    retriever_config = {
        "llm": llm,
        "vectorstore": vector_store,
        "metadata_field_info": metadata_field_info,
        "document_contents": "Resume information containing candidate details, skills, experience, and qualifications",
    }

    self_query_retriever = SelfQueryRetriever.from_llm(**retriever_config)

    k = kwargs.get("k", 15)
    include_justification = kwargs.get("include_justification", True)

    all_results = []
    try:
        results = self_query_retriever.invoke(query)
        if results:
            all_results.extend(results)
    except Exception as exc:
        print(f"Self-query retriever error: {exc}")

    candidate_names = []
    try:
        stored_metadata = vector_store.get(include=["metadatas"]).get("metadatas", [])
        for metadata in stored_metadata:
            candidate = metadata.get("candidate") or metadata.get("source") if metadata else None
            if candidate and candidate not in candidate_names:
                candidate_names.append(candidate)
    except Exception as exc:
        print(f"Candidate metadata lookup failed: {exc}")

    if candidate_names:
        candidate_results = []
        for candidate in candidate_names:
            try:
                candidate_results.extend(
                    vector_store.similarity_search(
                        query,
                        k=min(3, k),
                        filter={"candidate": candidate},
                    )
                )
            except Exception as exc:
                print(f"Candidate search failed for {candidate}: {exc}")
        if candidate_results:
            all_results.extend(candidate_results)

    if not all_results:
        try:
            all_results = vector_store.similarity_search(query, k=k)
        except Exception as exc:
            print(f"Vector similarity search failed: {exc}")
            return f"The resume index could not be queried at the moment. Please try again with a valid AI model setup. Query: {query}"

    initial_sources = set()
    if all_results:
        for result in all_results:
            source = result.metadata.get("source", "Unknown") if hasattr(result, "metadata") else "Unknown"
            initial_sources.add(source)

    if not candidate_names and len(initial_sources) < 2:
        broad_searches = [
            "skills experience qualifications",
            "education background professional",
            "achievements accomplishments projects",
            "responsibilities duties work",
        ]

        for search_term in broad_searches:
            if len(initial_sources) >= 2:
                break
            try:
                additional_results = vector_store.similarity_search(search_term, k=5)
                for result in additional_results:
                    source = result.metadata.get("source", "Unknown") if hasattr(result, "metadata") else "Unknown"
                    if source not in initial_sources:
                        all_results.append(result)
                        initial_sources.add(source)
            except Exception as exc:
                print(f"Additional search error: {exc}")

    source_contents = {}
    seen_content = set()

    if isinstance(all_results, list):
        result_limit = max(k * 2, len(candidate_names) * 3)
        for result in all_results[:result_limit]:
            content = result.page_content if hasattr(result, "page_content") else str(result)
            source = result.metadata.get("source", "Unknown") if hasattr(result, "metadata") else "Unknown"
            content_hash = hash(content.strip())
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            if source not in source_contents:
                source_contents[source] = []

            if len(source_contents[source]) < 3:
                source_contents[source].append(content)

    if not source_contents:
        return f"No relevant documents found for query: {query}"

    documents_content = []
    for source, contents in source_contents.items():
        for i, content in enumerate(contents, 1):
            if len(contents) > 1:
                documents_content.append(f"[From: {source} - Part {i}]\n{content}")
            else:
                documents_content.append(f"[From: {source}]\n{content}")

    combined_docs = "\n\n---\n\n".join(documents_content)
    sources_list = ", ".join(source_contents.keys()) if source_contents else "Multiple candidates"
    num_sources = len(source_contents)

    analysis_prompt = f"""You are an HR recruiter analyzing candidate resumes.
Based on the following resume documents from {num_sources} candidate(s), answer the user's question comprehensively.
IMPORTANT: You MUST include relevant information from EVERY candidate. Do NOT skip any candidate.

USER QUESTION: {query}

RESUME DOCUMENTS (from {num_sources} candidate(s)):
{combined_docs}

CRITICAL RULES - FOLLOW STRICTLY:
1. Include information from EVERY candidate shown above ({sources_list})
2. For each point or finding, EXPLICITLY mention which candidate(s) it applies to
3. If comparing candidates, discuss all candidates, not just one
4. Organize findings by candidate or theme, but ensure all candidates are covered
5. Use bullet points with specific evidence from each resume
6. Highlight strengths, skills, and relevant experience for EACH candidate
7. NEVER ignore any candidate - if a candidate is shown, analyze their information too
8. If applicable, note differences and unique qualities of each candidate

ANSWER:"""

    try:
        response = llm.invoke(analysis_prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        print(f"LLM analysis error: {exc}")
        return combined_docs


def parse_analysis_report(analysis_text):
    result = {
        "score": None,
        "recommendation": "Unknown",
        "reasoning": [],
    }

    lines = analysis_text.split("\n")

    for i, line in enumerate(lines):
        if "SUITABILITY SCORE" in line.upper():
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip():
                    score_text = lines[j].strip()
                    numbers = re.findall(r"\d+", score_text)
                    if numbers:
                        try:
                            result["score"] = int(numbers[0])
                            break
                        except ValueError:
                            pass
            if result["score"] is not None:
                break

    if result["score"] is None:
        score_patterns = [
            r"[Ss]uitability\s+score[:\s]+([\d.]+)",
            r"[Ss]core[:\s]+([\d]+)",
            r"(?:Score|Suitability)\s*[:=]\s*(\d+)",
        ]
        for pattern in score_patterns:
            match = re.search(pattern, analysis_text)
            if match:
                try:
                    result["score"] = int(float(match.group(1)))
                    break
                except (ValueError, IndexError):
                    pass

    for i, line in enumerate(lines):
        if "FINAL RECOMMENDATION" in line.upper() or "RECOMMENDATION" in line.upper():
            for j in range(i + 1, min(i + 5, len(lines))):
                rec_text = lines[j].strip().upper()
                if rec_text:
                    if "YES" in rec_text:
                        result["recommendation"] = "Yes"
                        break
                    elif "NO" in rec_text:
                        result["recommendation"] = "No"
                        break
            if result["recommendation"] != "Unknown":
                break

    if result["recommendation"] == "Unknown":
        for line in lines[-5:]:
            line_upper = line.strip().upper()
            if line_upper in ["YES", "NO", "YES.", "NO."]:
                result["recommendation"] = "Yes" if "YES" in line_upper else "No"
                break

    if result["recommendation"] == "Unknown" and result["score"] is not None:
        result["recommendation"] = "Yes" if result["score"] >= 70 else "No"

    for i, line in enumerate(lines):
        if "KEY" in line.upper() and "SKILL" in line.upper():
            for j in range(i + 1, min(i + 6, len(lines))):
                skill = lines[j].strip()
                if skill and len(skill) > 10 and not skill.startswith("5.") and not skill.startswith("4."):
                    result["reasoning"].append(skill)
                    if len(result["reasoning"]) >= 2:
                        break

        if len(result["reasoning"]) >= 2:
            break

    if not result["reasoning"]:
        bullets = re.findall(r"[-•*]\s+(.+?)(?:\n|$)", analysis_text)
        result["reasoning"] = [b.strip() for b in bullets[:2] if len(b.strip()) > 15]

    return result
