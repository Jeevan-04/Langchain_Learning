import io
import streamlit as st
from PyPDF2 import PdfReader
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            # Check if pdf is None
            if pdf is None:
                continue
            
            # Create a file-like object from the bytes
            pdf_file = io.BytesIO(pdf)
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except PdfReader.PdfReadError as e:  # Handle PdfReadError
            print(f"Error reading PDF: {e}")
            continue
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_stor(text_chunks):
    print("Creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    print("Writing index to file...")
    try:
        faiss.write_index_binary(vector_store, "faiss_index")
        print("Index written successfully.")
    except Exception as e:
        print(f"Error writing index: {e}")

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer and if the question asked is about your identity then say "I was created by Jeevan Naidu as part of Project NIRUKTI" and don't say anything about gemini and google\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

import faiss

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if the index file exists
    index_file = "faiss_index"
    if not os.path.exists(index_file):
        st.error(f"Faiss index file '{index_file}' does not exist.")
        return

    # Load the FAISS index
    index = faiss.read_index(index_file)

    # Assuming `similarity_search` method exists for your index
    docs = index.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("NIRUKTI")
    st.header("Chat with NIRUKTI")

    user_question = st.text_input("Ask a Question ")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_stor(text_chunks)
                    st.success("Done")
                else:
                    st.warning("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()