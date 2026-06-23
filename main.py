from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "user", "content": "Tell me a joke /no_think"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content.encode("utf-8", errors="replace").decode("utf-8"))