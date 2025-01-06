import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from io import BytesIO
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

st.set_page_config("Chat Multiple PDF")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key = GOOGLE_API_KEY)


# first concatenate all the text from all the documents
def get_pdf_text(pdf_docs):
    text=""
    for doc in pdf_docs:
        pdf_file = BytesIO(doc.read())
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text


# Then cut the concatenated text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)

    return text_chunks

# now we have chunks, so we need to conver these into vector embeddings
def get_vector_store(text_chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectore_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vectore_store.save_local("faiss_index")

# 
def get_conversational_chain():
    prompt_template="""
    Answer the question asked from the provided context. If the answer does not exist in the provided context then reply "Out of syllabus"
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:

    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Now we create the backend. all the above are helper functions but we need to define the flow of these functions in a proper function
# which will get executed through the frontend

def get_user_input(user_question):
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents":docs, "question":user_question})
    print(response)
    st.write("Answer:", response["output_text"])
 

def main():
    # st.set_page_config("Chat Multiple PDF")
    st.header("Chat with Multiple PDFs using Gemini")

    user_question = st.text_input("Ask a question from the PDF Files")

    if user_question:
        get_user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF here", accept_multiple_files=True, type="pdf")
        if st.button("Submit and Process"):
            with st.spinner("processing"):
                text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()