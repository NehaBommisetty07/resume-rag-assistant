# 📄 Resume RAG Assistant

An AI-powered Resume Question Answering Assistant built using Retrieval-Augmented Generation (RAG). Upload any resume in PDF format and ask natural language questions about the candidate.

## 🚀 Live Demo

👉https://resume-rag-assistant-rtpppktwtddpn9jypg9shx.streamlit.app/

## 📂 GitHub Repository

👉 https://github.com/NehaBommisetty07/resume-rag-assistant

---

## Features

- Upload PDF resumes
- Extract text from resumes
- Semantic search using FAISS
- Hugging Face Embeddings
- Context-aware responses using Groq Llama 3.3
- Streamlit web interface
- Handles missing information gracefully

---

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Groq API
- PyPDF
- dotenv

---

## Project Architecture

```
User Uploads Resume
        │
        ▼
 PDF Text Extraction
        │
        ▼
 Text Chunking
        │
        ▼
 HuggingFace Embeddings
        │
        ▼
 FAISS Vector Database
        │
        ▼
 User Question
        │
        ▼
 Similar Chunk Retrieval
        │
        ▼
 Groq Llama 3.3
        │
        ▼
 Final Answer
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/NehaBommisetty07/resume-rag-assistant.git
```

Move into the project

```bash
cd resume-rag-assistant
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GROQ_API_KEY=your_api_key_here
```

Run the application

```bash
streamlit run app.py
```

---

## Sample Questions

- What skills does the candidate have?
- Summarize the work experience.
- What projects are mentioned?
- What certifications are listed?
- What programming languages does the candidate know?
- What databases has the candidate worked with?

---

## Screenshots
<img width="1891" height="879" alt="image" src="https://github.com/user-attachments/assets/57bbb5a8-bcbf-4668-89c6-d7053f0d6d48" />
<img width="1906" height="874" alt="image" src="https://github.com/user-attachments/assets/5afebfaa-19d1-4e44-ad05-354adb9eed0d" />
<img width="1876" height="862" alt="image" src="https://github.com/user-attachments/assets/bc2e02c9-2e17-4c3d-a6eb-6b39ef88ee26" />


---

## Future Improvements

- ATS Resume Score
- Resume vs Job Description Matching
- Multi-resume comparison
- Chat history
- Source citations
- Docker deployment

---

## Author

**Neha Bommisetty**

GitHub:
https://github.com/NehaBommisetty07

LinkedIn:
https://www.linkedin.com/in/nehabommisetty/
