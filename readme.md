
# Retrieval-Augmented Generation Program

## Overview
This program implements a Retrieval-Augmented Generation (RAG) system, utilizing advanced natural language processing techniques to enhance the generation of text. It's built on the `langchain` framework and integrates various components for document loading, embeddings, and conversational interfaces.

## File Descriptions

### `embeddings.py`
This file handles the embedding functionalities required for the RAG system. It imports necessary modules from `langchain`, such as `OpenAIEmbeddings` and `Chroma` for embedding computations. It also includes text splitters and document loaders for various file formats (PDF, DOCX, TXT), which are crucial for processing input documents.

### `main.py`
This is the main executable file for the RAG system. It sets up a web interface using Streamlit, configuring the necessary callbacks, memories, and runnable configurations for the RAG system. It includes integration with the `langchain` and `langsmith` libraries for operational flow and user interface elements.

## Setup and Requirements
- Python 3.x
- Dependencies: `langchain`, `streamlit`, `langsmith`, and other supporting libraries.
- Installation: Use `pip install -r requirements.txt` to install the necessary packages.
  
## Usage
To run the program, navigate to the directory containing `main.py` and execute the command `streamlit run main.py`. This will start a local web server and the interface can be accessed through a web browser.

## Credits
This project builds upon the original work by Charly Wargnier, which can be found at [Charly Wargnier's LangchainRAG-Trubrics-Langsmith repository on GitHub](https://github.com/CharlyWargnier/LangchainRAG-Trubrics-Langsmith).