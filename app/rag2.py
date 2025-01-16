from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
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

# 履歴書を読み込んで文字列分割
file_path_resume = "resume.pdf"
loader_resume = PyPDFLoader(file_path_resume)
docs_resume = loader_resume.load() # pdf読み込み
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=200, add_start_index=True
)
all_splits_resume = text_splitter.split_documents(docs_resume)

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

# 情報取得処理を定義
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# toolを使用してデータ取得するか、直接応答するかをllmが判断し実行
# ユーザーのメッセージからデータ取得が必要と判断した場合、retrieveが実行される
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# retrieveによる情報を取得
tools = ToolNode([retrieve])

# retrieveで取得したデータを元に回答を生成
def generate(state: MessagesState):
    """Generate answer."""
    # 「toolを使用して取得したデータを元に生成したメッセージ」を最新のツールメッセージとして取得
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # 直前までのツールメッセージを含めたプロンプト生成
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# LangGraph構築
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# 実行
input_message = "履歴書の人物がJストリームで活かせそうな強みはなんでしょうか？「履歴書の人物のどのような点がJストリームのどの部分と合致し、どのように活かせそうか」を採用者視点で5つまで列挙してください。"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()