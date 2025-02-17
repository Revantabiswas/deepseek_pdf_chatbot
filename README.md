# deepseek_pdf_chatbot

This is a **Streamlit-based chatbot** designed to assist users in analyzing and querying research documents. Built with **LangChain** and powered by **Ollama's Deepseek models**, this chatbot allows users to upload PDF documents, process them, and ask questions to extract precise and concise answers. The application leverages advanced natural language processing (NLP) techniques, including text chunking, vector embeddings, and similarity search, to provide accurate responses based on the uploaded document's content.

## Key Features:
- **PDF Document Processing**: Upload and analyze PDF documents with ease.
- **Context-Aware Responses**: The chatbot uses the provided document context to generate accurate and relevant answers. 
- **Efficient Text Chunking**: Utilizes recursive text splitting for optimal document processing.
- **Ollama Integration**: Powered by Ollama's `deepseek-r1:1.5b` model for embeddings and language generation.
- **In-Memory Vector Store**: Fast and efficient document indexing and retrieval.

## How It Works:
1. Upload a PDF document containing research or text content.
2. The document is processed, chunked, and indexed using Ollama embeddings.
3. Ask questions about the document, and the chatbot will provide concise, context-based answers.

## Technologies Used:
- **Streamlit**: For building the interactive web application.
- **LangChain**: For document loading, text splitting, and prompt templating.
- **Ollama**: For embeddings (`deepseek-r1:1.5b`) and language model integration.
- **PDFPlumber**: For extracting text from PDF documents.

## Usage:
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the Streamlit app: `streamlit run app.py`.
4. Upload a PDF document and start asking questions!

## Example Use Cases:
- Research paper analysis.
- Document summarization.
- Quick fact-checking and information retrieval.

## Screenshots:
![Screenshot (1)](screenshots/Screenshot (1).png) 

## Contributions:
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

This description highlights the key features, technologies, and functionality of your chatbot, making it appealing to potential users and contributors. You can customize it further based on your preferences!
 
