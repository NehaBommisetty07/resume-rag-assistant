import streamlit as st

from rag_utils import (
    load_pdf,
    split_text,
    create_vectorstore,
    generate_answer,
)

st.set_page_config(
    page_title="Resume RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 800px; }
    .stChatMessage { border-radius: 12px; }
    h1 { font-size: 2rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 Resume RAG Assistant")
st.caption("Upload a resume and ask questions about it — powered by Groq Llama 3.3 70B.")

if "resume_processed" not in st.session_state:
    st.session_state.resume_processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

with st.sidebar:
    st.header("📤 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF)",
        type=["pdf"]
    )

    if uploaded_file is not None:
        st.success(f"✅ {uploaded_file.name} uploaded")

        if st.button("Process Resume", use_container_width=True, type="primary"):
            with st.spinner("Reading PDF and building the search index..."):
                try:
                    text = load_pdf(uploaded_file)
                    chunks = split_text(text)
                    vectorstore = create_vectorstore(chunks)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.resume_processed = True
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.chat_history = []

                    st.success("Resume processed! Ask a question below. 👇")
                except Exception as e:
                    st.session_state.resume_processed = False
                    st.error(f"❌ {e}")

    st.divider()

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.resume_processed = False
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.uploaded_filename = None
        st.rerun()

    if st.session_state.resume_processed:
        st.info(f"Active resume: **{st.session_state.uploaded_filename}**")

if not st.session_state.resume_processed:
    st.info("👈 Upload a resume PDF and click **Process Resume** to get started.")
else:
    for msg in st.session_state.chat_history:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the resume...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    answer = generate_answer(st.session_state.vectorstore, question)
                except Exception as e:
                    answer = f"⚠️ Something went wrong: {e}"
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})