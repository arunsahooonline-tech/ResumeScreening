# Resume Screening Suite 📋

An AI-powered resume screening and analysis application built with Streamlit. Automatically analyze multiple resumes against job descriptions using semantic search and advanced language models.

## Features ✨

- **Multi-Resume Upload**: Process multiple PDF and DOCX resume files simultaneously
- **Job Description Matching**: Compare resumes against target job descriptions
- **AI-Powered Analysis**: Extract key information and generate comprehensive screening reports
- **Semantic Search**: Ask natural language questions about resume content
- **Vector Database**: Fast and efficient resume retrieval using Chroma vector store
- **Multiple LLM Support**: Choose between Groq and Google Generative AI models
- **Interactive UI**: Step-by-step workflow with progress tracking
- **Beautiful Design**: Modern, responsive Streamlit interface with custom styling

## Tech Stack 🛠️

- **Framework**: [Streamlit](https://streamlit.io/)
- **LLMs**: 
  - [Groq](https://groq.com/) (recommended for free tier)
  - [Google Generative AI](https://ai.google.dev/)
- **Vector Database**: [Chroma](https://www.trychroma.com/)
- **Document Processing**: PyPDF, Docx2txt
- **LangChain**: For LLM orchestration and retrieval
- **Embeddings**: Google Generative AI Embeddings

## Installation 🚀

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd ResumeScreening
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirement.txt
   ```

## Configuration ⚙️

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# LLM Provider: "groq" or "google"
LLM_PROVIDER=groq

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# Google Configuration
GOOGLE_API_KEY=your_google_api_key_here

# LLM Model Configuration
LLM_MODEL=mixtral-8x7b-32768  # For Groq (example)
# OR
LLM_MODEL=gemini-pro  # For Google (example)

# Temperature (0-1): Higher = more creative, Lower = more focused
LLM_TEMPERATURE=0.3

# Embedding Model (Google only)
EMBEDDING_MODEL=gemini-embedding-001
```

### Obtaining API Keys

#### Groq API Key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up/login to your account
3. Navigate to API keys section
4. Create and copy your API key

#### Google API Key
1. Visit [Google AI Studio](https://aistudio.google.com)
2. Click "Get API Key"
3. Create a new API key for your project
4. Copy and save it securely

## Usage 📖

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Workflow

1. **Welcome Screen**: Overview of the application features
2. **Upload & Analyze**:
   - Enter the target job description
   - Upload one or multiple resume files (PDF or DOCX)
   - Click "Process" to analyze
3. **Query Results**: Ask natural language questions about the analyzed resumes

### Example Queries
- "What are the top 5 programming languages mentioned?"
- "Who has experience with machine learning?"
- "Find candidates with 5+ years of experience"
- "List all candidates who know Python and have project management experience"

## Project Structure 📁

```
ResumeScreening/
├── app.py                    # Main Streamlit application
├── resume_processer.py       # Resume processing and analysis logic
├── requirement.txt           # Python dependencies
├── README.md                 # This file
├── .env.example              # Example environment configuration
├── .venv/                    # Virtual environment (created during setup)
└── chroma_store/             # Vector database storage
    └── chroma.sqlite3        # Chroma database file
```

## File Descriptions 📄

### app.py
The main Streamlit application containing:
- Page configuration and custom CSS styling
- Three-step workflow UI (Welcome → Upload & Analyze → Query Results)
- Session state management
- File upload and processing interface
- Results display and query interface

### resume_processer.py
Core resume processing module with functions for:
- **load_resume()**: Load resume files (PDF, DOCX, TXT)
- **analyze_resume()**: AI-powered resume analysis against job description
- **create_vector_store()**: Build Chroma vector database from resume documents
- **run_self_query()**: Execute semantic search queries on resumes
- **parse_analysis_report()**: Parse and format analysis results

## Features Explained 🔍

### Resume Analysis
When you upload resumes and provide a job description, the application:
1. Extracts text from resume files
2. Chunks the text for better processing
3. Creates vector embeddings for semantic understanding
4. Uses the LLM to generate a comprehensive analysis report
5. Stores results in the Chroma vector database

### Semantic Search
The self-query retriever allows you to ask natural language questions that:
- Are converted into semantic search queries
- Find relevant resume sections
- Generate contextual answers using the LLM
- Support complex filtering and matching

## Troubleshooting 🐛

### Common Issues

**"API Key Error"**
- Ensure your `.env` file is in the project root
- Verify API keys are correct and have proper permissions
- Check that keys haven't been revoked or expired

**"No module named 'streamlit'"**
- Activate your virtual environment
- Run `pip install -r requirement.txt`

**"Unsupported file format"**
- Only PDF and DOCX files are supported
- Ensure files aren't corrupted

**"Vector store not found"**
- Process at least one resume first
- Check that `chroma_store/` directory exists and has proper permissions

## Performance Tips 📊

- **Batch Processing**: Process multiple resumes at once for efficiency
- **Job Description**: Provide detailed job descriptions (50+ characters recommended)
- **Temperature Setting**: Reduce temperature (0.1-0.3) for more consistent results
- **Model Selection**: Groq often provides faster responses than Google GenAI

## API Quotas & Limits

### Groq (Recommended)
- Free tier: Generous monthly quota
- Rate limit: Varies by model
- No credit card required for free tier

### Google GenAI
- Free tier: Limited daily quota
- Requires valid payment method on account
- Higher quotas available with paid plans

## Future Enhancements 🚀

- [ ] Batch resume ranking and scoring
- [ ] Resume template extraction and normalization
- [ ] Skill gap analysis and recommendations
- [ ] Export results as PDF reports
- [ ] Support for LinkedIn profiles
- [ ] Candidate comparison dashboard
- [ ] Advanced filtering and sorting options

## Contributing 🤝

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests with improvements
- Improve documentation

## License 📝

This project is part of the VariantsLab initiative. Please refer to the main repository for license information.

## Support & Contact 💬

For issues, questions, or suggestions:
- Check the troubleshooting section above
- Review the code documentation
- File an issue in the repository

---

**Happy Resume Screening! 🎉**

Built with ❤️ using Streamlit and AI
