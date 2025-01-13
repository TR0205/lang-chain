import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

@app.get("/")
def read_root():
    client = OpenAI()
    client.api_key = os.environ["OPENAI_API_KEY"]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        store=True,
        messages=[
            {"role": "user", "content": "こんにちは。今日の日本の天気は？"}
        ]
    )
    print(completion.choices[0].message)

    return completion.choices[0].message
