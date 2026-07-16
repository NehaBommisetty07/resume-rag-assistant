import os
import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# ==========================================================
# Cached Models
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def get_llm():

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found in .env"
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512
    )


# ==========================================================
# Read PDF
# ==========================================================

def load_pdf(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():
            raise ValueError(
                "No readable text found in PDF."
            )

        return text

    except Exception as e:
        raise RuntimeError(f"Failed to read PDF : {e}")


# ==========================================================
# Split Resume
# ==========================================================

def split_text(text):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,
        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            ", ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


# ==========================================================
# Build Vector Store
# ==========================================================

def create_vectorstore(chunks):

    if not chunks:
        raise ValueError("No chunks generated.")

    embedding_model = get_embedding_model()

    vectorstore = FAISS.from_texts(

        texts=chunks,

        embedding=embedding_model

    )

    return vectorstore


# ==========================================================
# Retrieve Documents
# ==========================================================

def retrieve_documents(vectorstore, question, k=5):

    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={

            "k": k,

            "fetch_k": 10,

            "lambda_mult": 0.7

        }

    )

    docs = retriever.invoke(question)

    return docs


# ==========================================================
# Prompt
# ==========================================================

def create_prompt(context, question):

    return f"""
You are an expert AI Resume Assistant.

Answer ONLY using the information provided in the resume.

Rules

1. Never make up information.

2. Never assume anything.

3. Use only resume content.

4. If responsibilities are asked,
summarize the work described.

5. If skills are asked,
collect them from every section.

6. If education is asked,
summarize it.

7. If projects are asked,
explain them.

8. Keep answers concise.

9. Use bullet points whenever possible.

10. Preserve dates,
company names,
project names,
skills,
technologies exactly.

11. If the answer truly does not exist reply EXACTLY:

"I couldn't find that information in the resume."

Resume

---------------------

{context}

---------------------

Question

{question}

Professional Answer
"""

# ==========================================================
# Generate Final Answer
# ==========================================================

def generate_answer(vectorstore, question):
    """
    Retrieves relevant chunks and generates a professional answer.
    """

    # Handle greetings without searching the resume
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if question.lower().strip() in greetings:
        return "Hello! 👋 Ask me anything about the uploaded resume."

    # -----------------------------
    # Query Expansion
    # -----------------------------
    query = question.lower()

    if "responsibil" in query:
        question += " duties work experience activities role"

    elif "project" in query:
        question += " developed built implementation"

    elif "skill" in query:
        question += " programming languages technologies tools"

    elif "database" in query:
        question += " MySQL SQL"

    elif "education" in query:
        question += " degree university college CGPA"

    elif "experience" in query:
        question += " internship work activities"

    elif "certification" in query:
        question += " certificates completed courses"

    try:

        docs = retrieve_documents(
            vectorstore,
            question
        )

        if not docs:
            return "I couldn't find that information in the resume."

        # Remove duplicate chunks
        context = "\n\n".join(
            dict.fromkeys(
                doc.page_content
                for doc in docs
            )
        )

        # Uncomment this while debugging retrieval
        # print("=" * 60)
        # print(context)
        # print("=" * 60)

        prompt = create_prompt(
            context,
            question
        )

        llm = get_llm()

        response = llm.invoke(prompt)

        answer = response.content.strip()

        if answer == "":
            return "I couldn't find that information in the resume."

        return answer

    except Exception as e:
        raise RuntimeError(
            f"Failed to generate answer: {e}"
        )