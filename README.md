# LocalRAG

## Overview

LocalRAG is an application designed to leverage the RAG (Retrieval-Augmented Generation) model to provide answers to questions based on a given context. This README file provides a brief overview of the application and its functionalities. The important is the Ollama integration. 

## Ollama
Ollama refers to a software/project tool that allows you to run large language models (LLMs) on your local machine. Here's a breakdown of what Ollama does:

- Runs Open-source LLMs: Ollama focuses on running open-source LLMs, such as Llama 2, which is a powerful language model itself.
- Local Execution: One of its key features is that it enables you to run these LLMs directly on your computer, without relying on cloud-based services.
- Simplified Setup: Ollama simplifies the setup and configuration process for running LLMs, including optimizing GPU usage for better performance.
- More info: https://ollama.com/ or https://github.com/ollama/ollama

## Installation

To use LocalRAG, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/arapellis-odysseas/localRag
    ```

2. Navigate to the cloned directory:

    ```
    cd LocalRAG
    ```

3. Create a virtual environment (optional but recommended):

    ```
    python3 -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:
    
    ```
    venv\Scripts\activate
    ```
    
    - On macOS and Linux:
    
    ```
    source venv/bin/activate
    ```

5. Install the required dependencies:

    ```
    pip3 install -r requirements.txt


## Functionality
The main.py file includes the following functionalities:

- Document Loading: Utilizes various document loaders from langchain_community to load text documents from the specified directory (data folder).

- Document Splitting: Splits the loaded documents into smaller chunks using the CharacterTextSplitter.

- Embedding Conversion and Storage: Converts document chunks into embeddings using the RAG model and stores them using Chroma.

## Question-Answering with RAG:

- Formats the documents for answering questions using a provided template.
-Accepts user input questions and provides answers based on the context using the RAG model.

## Configuration
You can customize the behavior of LocalRAG by modifying parameters within the main.py file. For instance, you can change the RAG model being used, adjust text chunk sizes, or modify file paths.

**IMPORTANT NOTE**

***Make sure that Ollama is installed and running in your machine***

## Usage

### Running the Application

LocalRAG consists of a single Python file (`main.py`) and a folder (`data`) containing text documents. To run the application, execute the following command:

1. Run the application:
    ```
    python main.py
    ```
2. Input your question when prompted. Type 'exit' to quit the application.
