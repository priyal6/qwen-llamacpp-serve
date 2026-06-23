# import time 
# import json
# import requests
# from statistics import mean

# BASE_URL = "http://localhost:8080"
# MODEL="local"

# PROMPTS = [
#     "Explain RAG in simple terms.",
#     "Write a short Python function to calculate factorial.",
#     "Summarize why quantized GGUF models are useful for local inference.",
#     "Explain overfitting in machine learning.",

# ]

# def estimate_tokens(text: str) -> int:
#     return max(1, len(text) // 4)

# def  measure_streaming_response(prompt: str):
#     url = f"{BASE_URL}/v1/chat/completions"

#     payload = {
#         "model": MODEL,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.2,
#         "max_tokens": 300,
#         "stream": True
#     }

#     start_time = time.perf_counter()
#     first_token_time = None
#     full_response = ""

#     response = requests.post(url, json=payload, stream=True)

#     for line in response.iter_lines():
#         if not line:
#             continue

#         decoded = line.decode("utf-8")

#         if decoded.startswith("data: "):
#             decoded = decoded.replace("data: ", "")
#         if decoded.strip() == "[DONE]":
#             break

#         try:
#             data = json.loads(decoded)
#             delta = data["choices"][0].get("delta", {})
#             token = delta.get("content", "")

#             if token:
#                 if first_token_time is None:
#                     first_token_time = time.perf_counter()

#                 full_response += token
#         except Exception:
#             continue

#     end_time = time.perf_counter()
#     input_tokens = estimate_tokens(prompt)
#     output_tokens = estimate_tokens(full_response)

#     total_latency = end_time - start_time
#     ttft = first_token_time - start_time if first_token_time else None
#     generation_time = end_time - first_token_time if first_token_time else total_latency

#     tokens_per_second = output_tokens/ generation_time if generation_time > 0 else 0 

#     return {
#         "prompt": prompt,
#         "response": full_response,
#         "input_tokens_est": input_tokens,
#         "output_tokens_est": output_tokens,
#         "ttft_seconds": ttft,
#         "total_latency_seconds": total_latency,
#         "generation_time_seconds": generation_time,
#         "output_tokens_per_second": tokens_per_second,
#     }

# def main():
#     results = []
#     for prompt in PROMPTS:
#         print("=" * 80)
#         print("Prompt:", prompt)

#         metrics = measure_streaming_response(prompt)
#         results.append(metrics)

#         print("Response:", metrics["response"][:300], "...")
#         print("Response:", metrics["response"][:300], "...")
#         print("Estimated input tokens:", metrics["input_tokens_est"])
#         print("Estimated output tokens:", metrics["output_tokens_est"])
#         # print("TTFT:", round(metrics["ttft_seconds"], 3), "sec")
#         print("Total latency:", round(metrics["total_latency_seconds"], 3), "sec")
#         print("Generation time:", round(metrics["generation_time_seconds"], 3), "sec")
#         print("Output tokens/sec:", round(metrics["output_tokens_per_second"], 2))

#     print("\n" + "=" * 80)
#     print("AVERAGE METRICS")
#     # print("Avg TTFT:", round(mean(r["ttft_seconds"] for r in results), 3), "sec")
#     print("Avg latency:", round(mean(r["total_latency_seconds"] for r in results), 3), "sec")
#     print("Avg output tokens/sec:", round(mean(r["output_tokens_per_second"] for r in results), 2))

# if __name__ == "__main__":
#     main()

import time
import requests

BASE_URL = "http://localhost:8080/completion"

prompts = [
    "Explain RAG in simple terms.",
    "Write a short Python function to calculate factorial."
]

for prompt in prompts:
    payload = {
        "prompt": prompt,
        "n_predict": 200,
        "temperature": 0.2,
        "stream": False
    }

    start = time.perf_counter()
    response = requests.post(BASE_URL, json=payload)
    end = time.perf_counter()

    print("=" * 80)
    print("Status:", response.status_code)
    print("Raw response:")
    print(response.text[:500])

    data = response.json()

    answer = data.get("content", "")
    latency = end - start

    completion_tokens = data.get("tokens_predicted", max(1, len(answer) // 4))
    prompt_tokens = data.get("tokens_evaluated", max(1, len(prompt) // 4))

    print("\nPrompt:", prompt)
    print("\nAnswer:")
    print(answer)

    print("\nMetrics:")
    print("Prompt tokens:", prompt_tokens)
    print("Completion tokens:", completion_tokens)
    print("Total latency:", round(latency, 3), "sec")
    print("Tokens/sec:", round(completion_tokens / latency, 2))