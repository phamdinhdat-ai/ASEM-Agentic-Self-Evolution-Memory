# pip install openai
from openai import OpenAI

client = OpenAI(
    api_key="sk-a7003232a711a6d441aa360c64eb54d57a71e42db3a0883e8631c38326b912c4",
    base_url="https://api.xah.io/v1",
)

response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[
        {"role": "user", "content": "Hello from CKEY"}
    ],
)

print(response.choices[0].message.content)