import os
from fastapi import FastAPI
from openai import OpenAI
from langchain_openai import ChatOpenAI

app = FastAPI()

@app.get("/openai")
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

@app.get("/lang")
def read_root():
    model = ChatOpenAI(model="gpt-3.5-turbo")
    res = model.invoke("Hello, World")
    return res

@app.get("/lang-trans")
def read_root():
    from langchain_core.messages import HumanMessage, SystemMessage
    # role指定
    messages = [
        SystemMessage("以下の言葉を日本語から英語に翻訳してください。"),
        HumanMessage("今日の朝ごはんは、納豆と味噌汁と卵焼きでした。")
    ]
    model = ChatOpenAI(model="gpt-3.5-turbo")
    res = model.invoke(messages)
    return res
    # {
    #     "content": "This morning's breakfast was natto, miso soup, and tamagoyaki.",
    #     "additional_kwargs": {
    #         "refusal": null
    #     },
    #     "response_metadata": {
    #         "token_usage": {
    #             "completion_tokens": 19,
    #             "prompt_tokens": 66,
    #             "total_tokens": 85,
    #             "completion_tokens_details": {
    #                 "accepted_prediction_tokens": 0,
    #                 "audio_tokens": 0,
    #                 "reasoning_tokens": 0,
    #                 "rejected_prediction_tokens": 0
    #             },
    #             "prompt_tokens_details": {
    #                 "audio_tokens": 0,
    #                 "cached_tokens": 0
    #             }
    #         },
    #         "model_name": "gpt-3.5-turbo-0125",
    #         "system_fingerprint": null,
    #         "finish_reason": "stop",
    #         "logprobs": null
    #     },
    #     "type": "ai",
    #     "name": null,
    #     "id": "run-fa2b23ad-9f54-45ac-acd9-78290cacccf9-0",
    #     "example": false,
    #     "tool_calls": [],
    #     "invalid_tool_calls": [],
    #     "usage_metadata": {
    #         "input_tokens": 66,
    #         "output_tokens": 19,
    #         "total_tokens": 85,
    #         "input_token_details": {
    #             "audio": 0,
    #             "cache_read": 0
    #         },
    #         "output_token_details": {
    #             "audio": 0,
    #             "reasoning": 0
    #         }
    #     }
    # }
