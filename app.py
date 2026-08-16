from groq import Groq
client = Groq()  # automatically reads GROQ_API_KEY from environment

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one word"}]
)
print(response.choices[0].message.content)