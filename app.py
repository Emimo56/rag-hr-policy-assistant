```python
import os
import io
import hashlib

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="wide"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

MODEL_NAME = "openai/gpt-oss-20b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 5


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# ---------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(pdf_file):
    """
    Extract text from every page of the uploaded PDF.
    """

    pdf_bytes = pdf_file.read()

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(pdf_document):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "page": page_number + 1,
                "text": text.strip()
            })

    pdf_document.close()

    return pages


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def create_chunks(pages):
    """
    Split extracted PDF text into overlapping chunks.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text.strip(),
                    "page": page_number
                })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ---------------------------------------------------------
# CREATE FAISS INDEX
# ---------------------------------------------------------

def create_faiss_index(chunks):
    """
    Convert chunks into embeddings and store them in FAISS.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# SEARCH RELEVANT CHUNKS
# ---------------------------------------------------------

def search_chunks(question, index, chunks, top_k=TOP_K):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):

        if index_number == -1:
            continue

        result = chunks[index_number].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ---------------------------------------------------------
# BUILD CONTEXT
# ---------------------------------------------------------

def build_context(search_results):

    context_parts = []

    for result in search_results:

        context_parts.append(
            f"[Page {result['page']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# GENERATE ANSWER
# ---------------------------------------------------------

def generate_answer(question, search_results):

    groq_client = get_groq_client()

    if not groq_client:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    context = build_context(search_results)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer employee questions using ONLY
the HR policy context provided by the application.

Rules:

1. Use only information contained in the supplied policy context.
2. Do not invent HR policies.
3. Do not use general knowledge when the answer is not in the policy.
4. If the answer cannot be found in the policy, clearly say:
   "I could not find this information in the uploaded HR policy."
5. Give a concise and professional answer.
6. When possible, mention the relevant policy page.
7. Do not make legal conclusions.
8. If the policy is ambiguous, explain the ambiguity instead
   of guessing.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}


EMPLOYEE QUESTION:

{question}


Answer the employee's question using only the HR policy context.
"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# DOCUMENT PROCESSING
# ---------------------------------------------------------

def process_document(uploaded_file):

    pages = extract_pdf_text(uploaded_file)

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "Could not create text chunks from the PDF."
        )

    index = create_faiss_index(chunks)

    return pages, chunks, index


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "pages" not in st.session_state:
    st.session_state.pages = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("📋 HR Policy Assistant")

st.write(
    "Upload an HR policy PDF and ask questions about "
    "leave, benefits, working hours, remote work, "
    "holidays, reimbursements, and other company policies."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Upload Policy")

    uploaded_file = st.file_uploader(
        "Upload an HR policy PDF",
        type=["pdf"]
    )

    st.divider()

    st.subheader("RAG Configuration")

    st.write(
        f"**Embedding model:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**LLM:** `{MODEL_NAME}`"
    )

    st.write(
        f"**Top K chunks:** `{TOP_K}`"
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    document_hash = hashlib.md5(file_bytes).hexdigest()

    if (
        st.session_state.document_hash
        != document_hash
    ):

        with st.spinner(
            "Processing HR policy..."
        ):

            try:

                # Reset file pointer
                uploaded_file.seek(0)

                pages, chunks, index = process_document(
                    uploaded_file
                )

                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.document_hash = document_hash
                st.session_state.chat_history = []

                st.success(
                    f"Policy processed successfully: "
                    f"{len(pages)} pages, "
                    f"{len(chunks)} chunks."
                )

            except Exception as error:

                st.error(
                    f"Error processing PDF: {error}"
                )


# ---------------------------------------------------------
# DOCUMENT STATUS
# ---------------------------------------------------------

if st.session_state.faiss_index is not None:

    st.success(
        "HR policy is ready. Ask a question below."
    )

    st.caption(
        f"Indexed {len(st.session_state.chunks)} "
        f"policy chunks."
    )

else:

    st.info(
        "👈 Upload an HR policy PDF from the sidebar "
        "to get started."
    )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# USER QUESTION
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about the HR policy..."
)


if question:

    if st.session_state.faiss_index is None:

        st.warning(
            "Please upload an HR policy PDF first."
        )

        st.stop()

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Search policy
    with st.spinner(
        "Searching the HR policy..."
    ):

        search_results = search_chunks(
            question,
            st.session_state.faiss_index,
            st.session_state.chunks
        )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    search_results
                )

                st.markdown(answer)

            except Exception as error:

                answer = (
                    "I was unable to generate an answer. "
                    f"Error: {error}"
                )

                st.error(answer)

        # Show sources
        if search_results:

            with st.expander(
                "📚 View retrieved policy sources"
            ):

                for number, result in enumerate(
                    search_results,
                    start=1
                ):

                    st.markdown(
                        f"**Source {number} — Page "
                        f"{result['page']}**"
                    )

                    st.caption(
                        f"Similarity score: "
                        f"{result['score']:.3f}"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
```
