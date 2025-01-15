import os
from fastapi import FastAPI
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

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

@app.get("/trans")
def read_root():
    from langchain_core.messages import HumanMessage, SystemMessage
    # role指定
    # ↓の書き方もOK
    # model.invoke([{"role": "user", "content": "Hello"}])
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

@app.get("/template")
def read_root():
    from langchain_core.prompts import ChatPromptTemplate

    model = ChatOpenAI(model="gpt-3.5-turbo")

    system_template = "以下の文章を日本語から{language}へ翻訳してください。"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"language": "英語", "text": "今日は目と頭が痛い"})

    return model.invoke(prompt)
    
@app.get("/pdf")
def read_root():
    from langchain_community.document_loaders import PyPDFLoader

    file_path = "test.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    return(
        len(docs), # ページ数
        f"{docs[1].page_content[:200]}\n", # 1ページ目の200文字目まで出力
        docs[1].metadata # ファイル名、ページ数
    )

@app.get("/split")
def read_root():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings

    file_path = "test.pdf"
    loader = PyPDFLoader(file_path)

    # pdf読み込み
    docs = loader.load()

    # chunk_size: 1000文字ずつに分割(チャンク)
    # chunk_overlap: 前後のチャンクで200文字重複する()
    # 理由：
    # - チャンクAの最後に「これは非常に重要なポイントです。」と書かれている。
    # - 次のチャンクBの最初にその「ポイント」が続いて説明されている。
    # 重複なしで分割すると意味が途切れてしまう可能性があるため
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # return all_splits[23], all_splits[24], all_splits[25]
    # return len(all_splits)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    # 同じモデルを使用すると基本的に同じ次元になる
    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    # return all_splits[0], all_splits[1], vector_1[:10]

    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)

    # pdfデータに基づく質問(同期)
    return vector_store.similarity_search(
        "2024年の経営方針は？"
    )

@app.get("/rag")
def read_root():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain import hub

    file_path_company = "jst.pdf"
    loader_company = PyPDFLoader(file_path_company)
    docs_company = loader_company.load() # pdf読み込み
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=200, add_start_index=True
    )
    all_splits_company = text_splitter.split_documents(docs_company)
    
    file_path_resume = "resume.pdf"
    loader_resume = PyPDFLoader(file_path_resume)
    docs_resume = loader_resume.load() # pdf読み込み
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=200, add_start_index=True
    )
    all_splits_resume = text_splitter.split_documents(docs_resume)

    documents = all_splits_company + all_splits_resume

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=documents)

    retrieved_docs = vector_store.similarity_search("履歴書または職務経歴書の人物が、株式会社Jストリームで活かせそうな強み")
    con = {"context": retrieved_docs}

    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in con["context"])
    messages = prompt.invoke({"question": "履歴書または職務経歴書の人物が、株式会社Jストリームで活かせそうな強みを箇条書きで出力してください", "context": docs_content})
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(messages)
    ans = {"answer": response.content}

    return ans
    