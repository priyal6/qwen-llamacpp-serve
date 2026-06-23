from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"

)

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

docs = [
    "RAG stands for Retrieval Augmented Generation.",
    "RAG retrieves relevant documents and gives them to the language model as context.",
    "RAG helps reduce hallucinations because the model answers using provided information.",
    "Qwen3-4B GGUF can be served locally using llama.cpp.",

]

doc_embeddings = embedder.encode(docs)

query = "How does RAG reduce hallucinations?"

query_embedding = embedder.encode([query])

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

best_doc_index = scores.argmax()
context = docs[best_doc_index]

prompt = f"""Use the context to answer the question.

Context:
{context}

Question:
{query}
/no_think"""

response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=200,
    temperature=0.2
)

print("Retrieved context:")
print(context)

print("\nAnswer:")
print(response.choices[0].message.content.encode("utf-8", errors="replace").decode("utf-8"))