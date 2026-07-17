import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


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
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_text(text)


# ---------------- Embeddings ---------------- #


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore


def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        max_new_tokens=512,
        temperature=0.3,
    )

    return ChatHuggingFace(llm=llm)


def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.5,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    st.title("PDF Chatbot")

    pdfs = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Process"):
        raw_text = get_pdf_text(pdfs)
        chunks = get_text_chunks(raw_text)

        vectorstore = get_vectorstore(chunks)

        st.success(f"Done! Created {len(chunks)} chunks.")
        st.write(vectorstore)

        llm = get_llm()
        response = llm.invoke("Say hello in one sentence.")
        st.write(response.content)


if __name__ == "__main__":
    main()
