import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents
def split_documents(documents, chunk_size=1800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
    
    chunks = text_splitter.split_documents(documents)
    

    return chunks
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating Gemini embeddings and storing in ChromaDB...")
        
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Starting ingestion pipeline...")
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)




if __name__ == "__main__":
    main()

