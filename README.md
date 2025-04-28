# Document Q&A Bot

This project demonstrates a simple Q&A bot that can answer questions based on the content of a PDF document. It utilizes LangChain for document loading, text splitting, embedding generation, and question answering. 

## Features

*   Loads PDF documents using `PyPDFLoader`.
*   Splits the document into smaller chunks using `RecursiveCharacterTextSplitter`.
*   Generates embeddings for each chunk using `HuggingFaceEmbeddings`.
*   Stores the embeddings in a FAISS vector store for efficient similarity search.
*   Uses a RetrievalQA chain to answer questions by finding relevant chunks and using a language model (Google PaLM 2 or OpenAI).

## Installation

1.  Clone this repository
git clone <repository-url>
2.  Install the required packages:
pip install -r requirements.txt
3.  Set up your API keys:
    *   **Google API Key:** Set the `GOOGLE_API_KEY` environment variable to your Google API key.
    *   **OpenAI API Key:** Set the `OPENAI_API_KEY` environment variable to your OpenAI API key if you plan to use OpenAI embeddings or models.

## Usage

1.  Run the Python script or Jupyter Notebook containing the code.
2.  Provide the path to your PDF document.
3.  Ask questions about the document content.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
