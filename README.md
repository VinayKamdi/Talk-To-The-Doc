# Talk to the Doc

**Talk to the Doc** is an interactive web application that allows users to upload PDF documents and ask questions about their content. The application leverages the power of language models to provide accurate responses based on the provided documents. It uses LangChain, FAISS, and Google Generative AI for document processing and retrieval.

## Features

- **PDF Upload**: Upload multiple PDF documents for analysis.
- **Document Embedding**: Convert documents into vector embeddings for efficient retrieval.
- **Question Answering**: Ask questions about the document content and receive accurate responses.
- **Language Models**: Utilizes advanced language models for natural language understanding and generation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/talk-to-the-doc.git
    cd talk-to-the-doc
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your environment variables. Create a `.env` file in the root directory and add your API keys:
    ```env
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your PDF documents using the file uploader.

4. Once the documents are uploaded, enter your questions in the provided text input box.

5. The application will process your query and display the answer based on the content of the uploaded documents.

## Code Overview

### Main Components

- **PDF Upload**: Allows users to upload multiple PDF files.
- **Vector Embedding**: Converts uploaded documents into vector embeddings using `GoogleGenerativeAIEmbeddings` and `FAISS`.
- **Question Answering**: Utilizes `ChatGroq` for generating responses based on the context extracted from documents.
- **Document Splitting**: Uses `RecursiveCharacterTextSplitter` to handle large documents by splitting them into manageable chunks.

### Key Functions

- `vector_embedding(files)`: Handles the conversion of uploaded PDF documents into vector embeddings.
- File Upload Handler: Manages the file uploading process and triggers the embedding function.
- Query Handling: Processes user questions and retrieves the most relevant answers from the document content.

## Dependencies

- `streamlit`
- `langchain_groq`
- `langchain.text_splitter`
- `langchain.chains`
- `langchain_core`
- `langchain_community`
- `langchain_google_genai`
- `faiss-cpu`
- `python-dotenv`

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, please contact [v1nayk4mdi@gmail.com].

---

Feel free to reach out if you have any questions or need further assistance. Enjoy using **Talk to the Doc**!
