from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from typing import Literal
from typing_extensions import Annotated

# 使用するLLMモデルを定義
llm = ChatOpenAI(model="gpt-3.5-turbo")
# llm = ChatOpenAI(model="gpt-4o-mini")

# 会社の魅力を読み込んで文字列分割
file_path_company = "jst.pdf"
loader_company = PyPDFLoader(file_path_company)
docs_company = loader_company.load() # pdf読み込み
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=200, add_start_index=True
)
all_splits_company = text_splitter.split_documents(docs_company)

# 文章を3分割する。ユーザーの検索クエリからLLMが文章中のどの部分を検索するか推定するために使用
total_documents = len(all_splits_company)
third = total_documents // 3
for i, document in enumerate(all_splits_company):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# 履歴書を読み込んで文字列分割
file_path_resume = "resume.pdf"
loader_resume = PyPDFLoader(file_path_resume)
docs_resume = loader_resume.load() # pdf読み込み
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=200, add_start_index=True
)
all_splits_resume = text_splitter.split_documents(docs_resume)

# 文章を3分割する。ユーザーの検索クエリからLLMが文章中のどの部分を検索するか推定するために使用
total_documents = len(all_splits_resume)
third = total_documents // 3
for i, document in enumerate(all_splits_resume):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# 分割結果を結合
documents = all_splits_company + all_splits_resume

# VectorDBへ追加
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=documents)

# RAG用プロンプトを指定
prompt = hub.pull("rlm/rag-prompt")

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# アプリケーションで扱うstate
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

# ユーザーの検索クエリを解析
# gpt-4o-miniでなければ動かない。3.5ではエラー
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

# 情報取得処理を定義
def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

# retrieveで取得した情報を元にllmが回答を返却
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# LangGraph構築
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# 実行
result = graph.invoke({"question": "履歴書の人物がJストリームで活かせそうな強みはなんでしょうか？採用者視点で5つまで列挙してください。"})

# print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')