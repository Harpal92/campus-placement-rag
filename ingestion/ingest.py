import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks: list[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings


def store_in_chromadb(chunks: list[str], embeddings, source_filename: str, collection_name: str = "placement_docs"):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )

    return collection


def retrieve_relevant_chunks(query: str, collection, top_k: int = 3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results["documents"][0]


if __name__ == "__main__":
    pdf_path = "data/Student Brochure_2027 Cognizant Ace Team program.pdf"

    text = extract_text_from_pdf(pdf_path)
    print(f"Total characters extracted: {len(text)}")

    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    embeddings = embed_chunks(chunks)
    print(f"Number of embeddings: {len(embeddings)}")

    filename_only = os.path.basename(pdf_path)
    collection = store_in_chromadb(chunks, embeddings, source_filename=filename_only)
    print(f"Stored {collection.count()} chunks in ChromaDB")

    query = "What is the CGPA requirement?"
    relevant_chunks = retrieve_relevant_chunks(query, collection, top_k=3)
    print(f"\nQuery: {query}")
    print("Top matching chunk(s):\n")
    for i, chunk in enumerate(relevant_chunks):
        print(f"--- Match {i+1} ---")
        print(chunk[:600])
        print()