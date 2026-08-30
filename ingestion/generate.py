from ingestion.ingest import retrieve_relevant_chunks
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-flash-latest")
def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant answering questions about campus placement documents.
Use ONLY the context below to answer the question. If the answer isn't in the context, say "I don't have that information in the provided documents."

Context:
{context}

Question: {query}

Answer:"""
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("placement_docs")

    #query = "What is the CGPA requirement?"
    query = "What is the CTC offered?"
    chunks = retrieve_relevant_chunks(query, collection, top_k=3)
    answer = generate_answer(query, chunks)
    print(answer)