from fastapi import FastAPI, UploadFile, File
import shutil
import os
import chromadb

from ingestion.ingest import (
    extract_text_from_pdf,
    chunk_text,
    embed_chunks,
    store_in_chromadb,
    retrieve_relevant_chunks,
)
from ingestion.generate import generate_answer

app = FastAPI()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "placement_docs"


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run your existing ingestion pipeline
    text = extract_text_from_pdf(temp_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    collection = store_in_chromadb(chunks, embeddings, source_filename=file.filename, collection_name=COLLECTION_NAME)

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks_in_db": collection.count()
    }


@app.post("/ask")
async def ask_question(query: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    chunks = retrieve_relevant_chunks(query, collection, top_k=3)
    answer = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": answer,
        "sources_used": chunks
    }