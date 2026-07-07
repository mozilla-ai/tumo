import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OTARI_API_KEY"],
    base_url="https://api.otari.ai/v1",
)

response = client.chat.completions.create(
    model="openai:gpt-4o",
    # model="mzai:moonshotai/Kimi-K2.6",
    messages=[{"role": "user", "content": "Say hello to TUMO folks"}],
)

print(response.choices[0].message.content)
