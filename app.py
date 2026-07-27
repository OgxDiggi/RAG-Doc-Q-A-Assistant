import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS

# ---------------- PDF ---------------- #


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


# ---------------- Splitter ---------------- #


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
    )
    return splitter.split_text(text)


# ---------------- Vector Store ---------------- #


@st.cache_resource(show_spinner=False)
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)


# ---------------- LLM ---------------- #


@st.cache_resource(show_spinner=False)
def get_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-8B",
        task="text-generation",
        huggingfacehub_api_token=st.secrets["HF_TOKEN"],
        max_new_tokens=512,
        temperature=0.3,
    )

    return ChatHuggingFace(llm=endpoint)


# ---------------- Main ---------------- #


def main():
    st.set_page_config(
        page_title="AI Document Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 AI Document Chatbot")
    st.caption("Upload PDFs and ask questions about them.")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("📄 Upload Documents")

        pdfs = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("📄 Process Documents"):
            if not pdfs:
                st.warning("Please upload at least one PDF.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()

            # Step 1
            status.text("📖 Reading PDF files...")
            progress.progress(20)
            raw_text = get_pdf_text(pdfs)

            # Step 2
            status.text("✂️ Splitting text into chunks...")
            progress.progress(45)
            chunks = get_text_chunks(raw_text)

            # Step 3
            status.text("🧠 Creating embeddings...")
            progress.progress(70)

            vectorstore = get_vectorstore(chunks)

            # Step 4
            status.text("💾 Building vector database...")
            progress.progress(90)

            st.session_state.vectorstore = vectorstore

            # Complete
            progress.progress(100)
            status.text("✅ Processing Complete!")

            st.success(
                f"Successfully processed {len(pdfs)} PDF(s) into {len(chunks)} searchable chunks."
            )

    # Display previous chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("First upload and process , then ask anything about the uploaded documents...")

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            if st.session_state.vectorstore is None:

                answer = "⚠️ Please upload and process PDFs first."

            else:

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 6,
                        "fetch_k": 20,
                    },
                )

                docs = retriever.invoke(prompt)

                context = "\n\n".join(doc.page_content for doc in docs)

                llm = get_llm()

                try:
                    with st.spinner("Generating answer..."):

                        response = llm.invoke(f"""
You are an intelligent AI assistant for document question answering.

Rules:
- Answer ONLY using the provided context.
- Do NOT use outside knowledge.
- If the answer is not present in the context, reply exactly:
  "I couldn't find that information in the uploaded documents."
- Keep the answer concise, accurate, and well-structured.

Context:
{context}

Question:
{prompt}

Answer:
""")

                    answer = response.content

                except Exception:
                    answer = "⚠️ Unable to generate a response. Please try again."

            st.markdown(answer)

            if st.session_state.vectorstore is not None:
                with st.expander("📚 Retrieved Context"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(doc.page_content[:500] + "...")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


if __name__ == "__main__":
    main()
