/// <reference path="./vite-env.d.ts" />
// @refresh reset

import React, {
  ChangeEvent,
  FormEvent,
  useEffect,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom/client'
import './style.css'

type ParsedResult = {
  score?: number | null
  recommendation?: string | object
  reasoning?: Array<string | object>
}

type StepState = [boolean, boolean, boolean]

const API_BASE = 'http://localhost:8000'

const acceptedFormats = '.pdf, .docx, .txt'

const defaultJobDescription =
  'We are hiring a Senior Python Developer with experience in backend APIs, AI systems, cloud computing, and team leadership.'

/**
 * Converts API/LLM values into safe text for React rendering.
 *
 * Handles:
 * - strings
 * - numbers
 * - booleans
 * - arrays
 * - Gemini/LangChain objects such as:
 *   { type: "text", text: "...", extras: {...} }
 * - nested content objects
 */
const toDisplayText = (value: unknown): string => {
  if (value === null || value === undefined) {
    return ''
  }

  if (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return String(value)
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => toDisplayText(item))
      .filter(Boolean)
      .join('\n')
  }

  if (typeof value === 'object') {
    const objectValue = value as Record<string, unknown>

    // Gemini/LangChain content object
    if (typeof objectValue.text === 'string') {
      return objectValue.text
    }

    // Some LLM providers return nested content
    if (objectValue.content !== undefined) {
      return toDisplayText(objectValue.content)
    }

    // Some responses may contain a message field
    if (objectValue.message !== undefined) {
      return toDisplayText(objectValue.message)
    }

    // Last resort
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }

  return String(value)
}

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const savedTheme = localStorage.getItem('resume-screening-theme')

    return savedTheme === 'light' || savedTheme === 'dark'
      ? savedTheme
      : 'dark'
  })

  const [currentStep, setCurrentStep] = useState(0)

  const [stepCompleted, setStepCompleted] = useState<StepState>([
    true,
    false,
    false,
  ])

  const [jobDescription, setJobDescription] = useState(
    defaultJobDescription,
  )

  const [selectedFiles, setSelectedFiles] = useState<File[]>([])

  const [analysis, setAnalysis] = useState('')

  const [parsed, setParsed] = useState<ParsedResult | null>(null)

  const [status, setStatus] = useState('Ready to analyze')

  const [loading, setLoading] = useState(false)

  const [query, setQuery] = useState(
    'Which candidate is the strongest fit for this role?',
  )

  const [queryAnswer, setQueryAnswer] = useState('')

  const [queryLoading, setQueryLoading] = useState(false)

  const [apiOnline, setApiOnline] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)

  /**
   * Check FastAPI health endpoint.
   */
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => {
        setApiOnline(response.ok)
      })
      .catch(() => {
        setApiOnline(false)
      })
  }, [])

  /**
   * Apply selected theme.
   */
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('resume-screening-theme', theme)
  }, [theme])

  /**
   * Handle resume file selection.
   */
  const handleFileChange = (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const newFiles = Array.from(event.target.files ?? [])

    setSelectedFiles((currentFiles) => {
      const files = [...currentFiles, ...newFiles]

      return files.filter(
        (file, index, allFiles) =>
          index ===
          allFiles.findIndex(
            (candidate) =>
              candidate.name === file.name &&
              candidate.size === file.size &&
              candidate.lastModified === file.lastModified,
          ),
      )
    })
  }

  /**
   * Analyze uploaded resumes.
   */
  const handleAnalyze = async (
    event: FormEvent,
  ) => {
    event.preventDefault()

    if (!selectedFiles.length) {
      setStatus('Please upload at least one resume file.')
      return
    }

    if (!jobDescription.trim()) {
      setStatus('Please enter a job description.')
      return
    }

    const formData = new FormData()

    formData.append(
      'job_description',
      jobDescription,
    )

    selectedFiles.forEach((file) => {
      formData.append('files', file)
    })

    setLoading(true)
    setStatus('Analyzing resumes...')
    setQueryAnswer('')

    try {
      const response = await fetch(
        `${API_BASE}/api/analyze`,
        {
          method: 'POST',
          body: formData,
        },
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ?? 'Analysis failed.',
        )
      }

      /**
       * IMPORTANT:
       *
       * data.analysis may be an object such as:
       *
       * {
       *   type: "text",
       *   text: "...",
       *   extras: {}
       * }
       *
       * React cannot render that object directly.
       *
       * Normalize it to a string first.
       */
      setAnalysis(
        toDisplayText(data.analysis),
      )

      setParsed(
        data.parsed ?? null,
      )

      setStepCompleted([
        true,
        true,
        true,
      ])

      setCurrentStep(2)

      setStatus(
        `Analysis complete for ${
          data.document_count ??
          selectedFiles.length
        } resume(s).`,
      )
    } catch (error) {
      setStatus(
        error instanceof Error
          ? error.message
          : 'Unable to analyze resumes.',
      )
    } finally {
      setLoading(false)
    }
  }

  /**
   * Ask a question about candidate data.
   */
  const handleQuery = async () => {
    if (!query.trim()) {
      setQueryAnswer(
        'Please enter a question to ask about the candidate data.',
      )

      return
    }

    setQueryLoading(true)
    setQueryAnswer('')

    try {
      const response = await fetch(
        `${API_BASE}/api/query`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: query,
          }),
        },
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ?? 'Query failed.',
        )
      }

      /**
       * data.answer may also be a Gemini/LangChain
       * content object instead of a string.
       */
      setQueryAnswer(
        data.answer !== undefined &&
        data.answer !== null
          ? toDisplayText(data.answer)
          : 'No answer returned.',
      )
    } catch (error) {
      setQueryAnswer(
        error instanceof Error
          ? error.message
          : 'Unable to query the data.',
      )
    } finally {
      setQueryLoading(false)
    }
  }

  /**
   * Reset complete workflow.
   */
  const resetWorkflow = () => {
    setCurrentStep(0)

    setStepCompleted([
      true,
      false,
      false,
    ])

    setAnalysis('')

    setParsed(null)

    setQueryAnswer('')

    setJobDescription(
      defaultJobDescription,
    )

    setStatus('Ready to analyze')

    setSelectedFiles([])

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  /**
   * Safely convert recommendation to text.
   */
  const recommendationText =
    parsed?.recommendation
      ? toDisplayText(
          parsed.recommendation,
        )
      : '—'

  /**
   * Display suitability score.
   */
  const suitabilityScore =
    parsed?.score !== null &&
    parsed?.score !== undefined
      ? `${parsed.score}/100`
      : '—'

  return (
    <div className="app-shell">

      {/* =====================================================
          SIDEBAR
          ===================================================== */}

      <aside className="sidebar panel">

        <div>
          <p className="eyebrow">
            Workflow
          </p>

          <h3>
            Recruitment flow
          </h3>
        </div>

        <div className="nav-stack">

          <button
            type="button"
            className={`nav-btn ${
              currentStep === 0
                ? 'active'
                : ''
            }`}
            onClick={() =>
              setCurrentStep(0)
            }
            aria-current={
              currentStep === 0
                ? 'step'
                : undefined
            }
          >
            <span>
              {stepCompleted[0]
                ? '✅'
                : '1'}
            </span>

            Welcome
          </button>

          <button
            type="button"
            className={`nav-btn ${
              currentStep === 1
                ? 'active'
                : ''
            }`}
            onClick={() =>
              setCurrentStep(1)
            }
            aria-current={
              currentStep === 1
                ? 'step'
                : undefined
            }
          >
            <span>
              {stepCompleted[1]
                ? '✅'
                : '2'}
            </span>

            Upload & Analyze
          </button>

          <button
            type="button"
            className={`nav-btn ${
              currentStep === 2
                ? 'active'
                : ''
            }`}
            onClick={() =>
              setCurrentStep(2)
            }
            disabled={!stepCompleted[1]}
            aria-current={
              currentStep === 2
                ? 'step'
                : undefined
            }
          >
            <span>
              {stepCompleted[2]
                ? '✅'
                : '3'}
            </span>

            Query Results
          </button>

        </div>

        <div className="progress-wrap">

          <div className="progress-label">
            Progress
          </div>

          <div className="progress-bar">
            <span
              style={{
                width: `${
                  (stepCompleted.filter(
                    Boolean,
                  ).length /
                    3) *
                  100
                }%`,
              }}
            />
          </div>

        </div>

        <button
          type="button"
          className="reset-btn"
          onClick={resetWorkflow}
        >
          Reset
        </button>

      </aside>

      {/* =====================================================
          MAIN PANEL
          ===================================================== */}

      <div className="main-panel">

        {/* ===================================================
            TOP BAR
            =================================================== */}

        <header className="topbar">

          <div>

            <p className="eyebrow">
              AI recruitment platform
            </p>

            <h1>
              Resume Screening Suite
            </h1>

          </div>

          <div className="topbar-actions">

            <button
              type="button"
              className="theme-switch"
              onClick={() =>
                setTheme(
                  (currentTheme) =>
                    currentTheme ===
                    'dark'
                      ? 'light'
                      : 'dark',
                )
              }
              aria-pressed={
                theme === 'light'
              }
              aria-label={`Switch to ${
                theme === 'dark'
                  ? 'light'
                  : 'dark'
              } theme`}
            >
              <span aria-hidden="true">
                {theme === 'dark'
                  ? '☀'
                  : '☾'}
              </span>

              {theme === 'dark'
                ? 'Light theme'
                : 'Dark theme'}
            </button>

            <div
              className={`status-pill ${
                apiOnline
                  ? 'online'
                  : 'offline'
              }`}
            >
              {apiOnline
                ? 'API online'
                : 'API offline'}
            </div>

          </div>

        </header>

        {/* ===================================================
            STEP 1 - WELCOME
            =================================================== */}

        {currentStep === 0 && (
          <section className="panel hero-panel">

            <div className="hero-box">

              <h2>
                AI-Powered Resume Screening
              </h2>

              <p>
                Analyze resumes and ask
                intelligent questions powered
                by AI.
              </p>

            </div>

            <div className="feature-grid">

              <div className="feature-card">
                <strong>
                  📤 Upload
                </strong>

                <span>
                  PDF & DOCX resumes
                </span>
              </div>

              <div className="feature-card">
                <strong>
                  🤖 Analyze
                </strong>

                <span>
                  AI-powered insights
                </span>
              </div>

              <div className="feature-card">
                <strong>
                  🔍 Query
                </strong>

                <span>
                  Ask any question
                </span>
              </div>

            </div>

            <button
              type="button"
              className="primary-btn"
              onClick={() =>
                setCurrentStep(1)
              }
            >
              Get Started
            </button>

          </section>
        )}

        {/* ===================================================
            STEP 2 - UPLOAD & ANALYZE
            =================================================== */}

        {currentStep === 1 && (
          <main className="workspace-grid">

            {/* ===============================================
                UPLOAD FORM
                =============================================== */}

            <form
              className="panel"
              onSubmit={handleAnalyze}
            >

              <div className="panel-header">

                <h2>
                  Upload & Analyze
                </h2>

                <span className="muted">
                  {status}
                </span>

              </div>

              <div className="input-group">

                <label htmlFor="job-description">
                  Job Description
                </label>

                <textarea
                  id="job-description"
                  value={jobDescription}
                  onChange={(event) =>
                    setJobDescription(
                      event.target.value,
                    )
                  }
                  placeholder="Paste the job specification here..."
                />

              </div>

              {/* =========================================
                  FILE UPLOAD
                  ========================================= */}

              <div className="upload-box">

                <div className="upload-label">
                  Upload resumes
                </div>

                <div className="helper-text">
                  Accepted formats:{' '}
                  {acceptedFormats}
                </div>

                <div className="upload-control">

                  <label
                    className="upload-trigger"
                    htmlFor="resume-upload"
                  >

                    <span
                      className="upload-icon"
                      aria-hidden="true"
                    >
                      ↑
                    </span>

                    <span>
                      {selectedFiles.length
                        ? 'Add more resumes'
                        : 'Choose resume files'}
                    </span>

                  </label>

                  <input
                    ref={fileInputRef}
                    id="resume-upload"
                    type="file"
                    multiple
                    accept=".pdf,.docx,.txt"
                    onChange={
                      handleFileChange
                    }
                  />

                  <span className="upload-count">

                    {selectedFiles.length
                      ? `${
                          selectedFiles.length
                        } file${
                          selectedFiles.length ===
                          1
                            ? ''
                            : 's'
                        } selected`
                      : 'No files selected'}

                  </span>

                </div>

                {selectedFiles.length >
                0 ? (
                  <div className="file-list">

                    {selectedFiles.map(
                      (file) => (
                        <span
                          key={`${file.name}-${file.size}`}
                          className="file-tag"
                        >
                          {file.name}
                        </span>
                      ),
                    )}

                  </div>
                ) : (
                  <div className="empty-file-list">
                    No resumes selected yet.
                  </div>
                )}

              </div>

              {/* =========================================
                  ANALYZE BUTTON
                  ========================================= */}

              <button
                className="primary-btn wide"
                type="submit"
                disabled={loading}
              >
                {loading
                  ? 'Analyzing...'
                  : 'Run analysis'}
              </button>

            </form>

            {/* ===============================================
                CANDIDATE SUMMARY
                =============================================== */}

            <aside className="panel summary-panel">

              <h2>
                Candidate Summary
              </h2>

              <div className="stat-grid">

                <div className="stat-card">

                  <span>
                    Suitability
                  </span>

                  <strong>
                    {suitabilityScore}
                  </strong>

                </div>

                <div className="stat-card">

                  <span>
                    Recommendation
                  </span>

                  <strong>
                    {recommendationText}
                  </strong>

                </div>

                <div className="stat-card">

                  <span>
                    Files
                  </span>

                  <strong>
                    {selectedFiles.length}
                  </strong>

                </div>

              </div>

              <div className="result-box">

                <h3>
                  Recruiter feedback
                </h3>

                <p>

                  {parsed?.reasoning?.length
                    ? parsed.reasoning
                        .map((item) =>
                          toDisplayText(
                            item,
                          ),
                        )
                        .join(' ')
                    : 'Upload resumes and run analysis to see AI recruiter feedback.'}

                </p>

              </div>

            </aside>

          </main>
        )}

        {/* ===================================================
            STEP 3 - QUERY RESULTS
            =================================================== */}

        {currentStep === 2 && (
          <section className="panel query-panel">

            <div className="panel-header">

              <h2>
                Query Results
              </h2>

              <span className="muted">
                Ask about candidate fit and
                strengths
              </span>

            </div>

            <div className="query-box">

              <textarea
                value={query}
                onChange={(event) =>
                  setQuery(
                    event.target.value,
                  )
                }
                placeholder="Ask about skills, experience, or fit..."
              />

              <button
                className="secondary-btn"
                type="button"
                onClick={handleQuery}
                disabled={queryLoading}
              >
                {queryLoading
                  ? 'Querying...'
                  : 'Ask'}
              </button>

              {queryAnswer && (
                <div className="query-answer">
                  {queryAnswer}
                </div>
              )}

            </div>

            {/* =============================================
                DETAILED ANALYSIS
                ============================================= */}

            {analysis && (
              <div className="analysis-panel">

                <h3>
                  Detailed Analysis
                </h3>

                <pre>
                  {analysis}
                </pre>

              </div>
            )}

          </section>
        )}

      </div>
    </div>
  )
}

/* =========================================================
   REACT ROOT
   ========================================================= */

const rootElement =
  document.getElementById('root')

if (!rootElement) {
  throw new Error(
    'Root element not found',
  )
}

const reactRoot =
  (
    rootElement as HTMLElement & {
      __resumeRoot?: ReturnType<
        typeof ReactDOM.createRoot
      >
    }
  ).__resumeRoot ??=
  ReactDOM.createRoot(
    rootElement,
  )

reactRoot.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)