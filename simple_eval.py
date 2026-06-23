
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


docs = [
    "RAG stands for Retrieval Augmented Generation.",
    "RAG retrieves relevant documents and gives them to the language model as context.",
    "RAG helps reduce hallucinations because the model answers using provided information.",
    "Qwen3-4B GGUF can be served locally using llama.cpp.",
]

doc_embeddings = embedder.encode(docs)


eval_questions = [
    {
        "question": "What does RAG stand for?",
        "expected_doc_index": 0,
        "expected_keywords": ["retrieval", "augmented", "generation"]
    },
    {
        "question": "How does RAG reduce hallucinations?",
        "expected_doc_index": 2,
        "expected_keywords": ["provided information", "context", "reduce hallucinations"]
    },
    {
        "question": "How can Qwen3-4B GGUF be run locally?",
        "expected_doc_index": 3,
        "expected_keywords": ["llama.cpp", "locally"]
    }
]


def retrieve(query):
    query_embedding = embedder.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    best_doc_index = scores.argmax()
    return best_doc_index, docs[best_doc_index]

def ask_llm(question, context):
    prompt = f"""Use the context to answer the question.

    Context:
    {context}

    Question:
    {question}
    /no_think"""


    response = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0
    )

    return response.choices[0].message.content

def keyword_score(answer, keywords):
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits/ len(keywords)

total_retrieval = 0
total_answer_score = 0

for item in eval_questions:
    question = item["question"]
    expected_doc_index = item["expected_doc_index"]
    expected_keywords = item["expected_keywords"]

    retrieved_index, context = retrieve(question)
    answer = ask_llm(question, context)

    retrieval_correct = retrieved_index == expected_doc_index
    answer_score = keyword_score(answer, expected_keywords)


    total_retrieval += int(retrieval_correct)
    total_answer_score += answer_score

    print("=" * 70)
    print("Question:", question)
    print("Retrieved Doc:", context)
    print("Expected Doc Index:", expected_doc_index)
    print("Retrieved Doc Index:", retrieved_index)
    print("Retrieval Correct:", retrieval_correct)
    print("Answer:", answer)
    print("Answer Keyword Score:", round(answer_score, 2))

print("=" * 70)
print("Final Retrieval Accuracy:", round(total_retrieval / len(eval_questions), 2))
print("Final Average Answer Score:", round(total_answer_score / len(eval_questions), 2))
