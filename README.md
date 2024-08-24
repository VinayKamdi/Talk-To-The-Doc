# Talk to the Doc

**Talk to the Doc** is an interactive web application that allows users to upload PDF documents, Word documents, and images to ask questions about their content. The application leverages advanced language models to provide accurate responses based on the provided documents. It uses LangChain, FAISS, and Google Generative AI for document processing and retrieval.

## Demo Link
https://talk-to-the-doc.streamlit.app/

## Features

- **Multi-format Upload**: Upload multiple PDF, DOCX, and image files for analysis.
- **Document Embedding**: Convert documents into vector embeddings for efficient retrieval.
- **Text Extraction**: Extract text from PDF, DOCX, and image files, including PDFs that consist of images.
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
    streamlit run talk_to_the_doc.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your documents (PDF, DOCX, or images) using the file uploader.

4. Once the documents are uploaded, enter your questions in the provided text input box.

5. The application will process your query and display the answer based on the content of the uploaded documents.

## Code Overview

### Main Components

- **File Upload**: Allows users to upload multiple PDF, DOCX, and image files.
- **Text Extraction**: Extracts text from various formats, including OCR for image-based PDFs and other image files.
- **Vector Embedding**: Converts uploaded documents into vector embeddings using `GoogleGenerativeAIEmbeddings` and `FAISS`.
- **Question Answering**: Utilizes `ChatGroq` for generating responses based on the context extracted from documents.
- **Document Splitting**: Uses `RecursiveCharacterTextSplitter` to handle large documents by splitting them into manageable chunks.

### Key Functions

- `extract_text_from_doc(file_path)`: Extracts text from DOCX files.
- `extract_text_from_image(image_path)`: Extracts text from image files using OCR.
- `extract_text_from_pdf_images(pdf_path)`: Converts each page of a PDF into an image and extracts text using OCR.
- `vector_embedding(files)`: Handles the conversion of uploaded documents into vector embeddings.

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
- `pdf2image`
- `pytesseract`
- `python-docx`
- `Pillow`
- `python-magic`

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, please contact [v1nayk4mdi@gmail.com].

---

Feel free to reach out if you have any questions or need further assistance. Enjoy using **Talk to the Doc**!
