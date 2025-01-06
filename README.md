
# Chat with Multiple PDFs using Generative AI (RAG)

## Description
This project implements a **Retrieval-Augmented Generation (RAG)** system to enable users to interact with and query multiple PDF documents. By combining retrieval techniques with generative AI, it delivers accurate, context-aware answers to user questions based on uploaded PDF content.

## Key Features
- **PDF Processing**:
  - Extracts text from user-uploaded PDF documents.
  - Handles multiple PDFs simultaneously.
- **Text Chunking**:
  - Splits extracted text into manageable chunks for embedding generation.
  - Uses `RecursiveCharacterTextSplitter` for efficient splitting with overlapping context.
- **Embeddings and Vector Storage**:
  - Converts text chunks into vector embeddings using `Google Generative AI (embedding-gecko-001)`.
  - Stores embeddings locally in a FAISS vector database for fast retrieval.
- **Conversational Chain**:
  - Retrieves relevant chunks from the FAISS database based on user queries.
  - Utilizes `gemini-pro` from Google Generative AI to generate context-aware responses.
- **Streamlit Frontend**:
  - User-friendly interface to upload PDFs, process them, and ask questions.
  - Sidebar for document upload and processing with real-time feedback.

## Technology Stack
- **Frontend**: Streamlit for user interaction.
- **Backend**:
  - `PyPDF2` for PDF text extraction.
  - `LangChain` for text chunking, embeddings, and conversational chain management.
  - `FAISS` for similarity search.
  - `Google Generative AI` for embedding and response generation.
- **Environment Management**:
  - `.env` file for secure API key management.
  - Python virtual environment for dependency isolation.

## How It Works
1. **Upload PDFs**: Users upload one or more PDF files through the Streamlit interface.
2. **Process Text**: The system extracts, chunks, and embeds the text.
3. **Query**: Users ask questions about the documents.
4. **Retrieve and Generate**:
   - Relevant chunks are retrieved from FAISS.
   - The generative AI model generates a context-based response.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
  ```
2. Create a virtual environment and install dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
3. Add your Google API key in a .env file:
  ```bash
  GOOGLE_API_KEY=<your-google-api-key>
  ```
4. Run the application:
  ```bash
  streamlit run app.py
  ```

