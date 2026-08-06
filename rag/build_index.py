import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = "data/corpus"
DB_DIR = "chroma_db"
COLLECTION_NAME = "southern_lk_corpus"


def load_documents():
    documents = []
    filenames = []
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(CORPUS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(text)
            filenames.append(filename)
    return documents, filenames


def chunk_documents(documents, filenames):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    all_chunks = []
    all_metadatas = []
    all_ids = []
    for doc_text, filename in zip(documents, filenames):
        chunks = splitter.split_text(doc_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename})
            all_ids.append(f"{filename}_chunk{i}")
    return all_chunks, all_metadatas, all_ids


def build_index():
    print("Loading documents from", CORPUS_DIR)
    documents, filenames = load_documents()
    print(f"Loaded {len(documents)} documents: {filenames}")

    print("Chunking documents...")
    chunks, metadatas, ids = chunk_documents(documents, filenames)
    print(f"Created {len(chunks)} chunks")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Adding chunks to collection (this embeds them automatically)...")
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Done. Indexed {collection.count()} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()