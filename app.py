from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain import hub
import os

modelList = ["gemma:2b",
             "llama2"]

model_local = ChatOllama(model=modelList[1])

# 1. Load the documents

# get the current directory absolute path
dir_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dir_path, "data")

text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()


# 2. Split the documents
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 3. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    persist_directory="chroma",
)
retriever = vectorstore.as_retriever()


# 4. After RAG
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("################")
after_rag_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
after_rag_prompt = PromptTemplate.from_template(after_rag_template)


while True:
    after_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
    )
    print("\n\n")
    question = input("Question (or 'exit' to quit): ")
    if question == "exit":
        break

    for chunk in after_rag_chain.stream(question):
        print(chunk, end="", flush=True)
