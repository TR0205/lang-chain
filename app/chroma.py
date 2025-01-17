from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from uuid import uuid4



llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("test_collection")
# collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=embeddings,
)

# 会社の魅力を読み込んで文字列分割
file_path_company = "jst.pdf"
loader_company = PyPDFLoader(file_path_company)
docs_company = loader_company.load() # pdf読み込み



# # 文章を3分割する。ユーザーの検索クエリからLLMが文章中のどの部分を検索するか推定するために使用
# total_documents = len(all_splits_company)
# third = total_documents // 3
# for i, document in enumerate(all_splits_company):
#     if i < third:
#         document.metadata["section"] = "beginning"
#     elif i < 2 * third:
#         document.metadata["section"] = "middle"
#     else:
#         document.metadata["section"] = "end"

# 履歴書を読み込んで文字列分割
file_path_resume = "resume.pdf"
loader_resume = PyPDFLoader(file_path_resume)
docs_resume = loader_resume.load() # pdf読み込み

# # 文章を3分割する。ユーザーの検索クエリからLLMが文章中のどの部分を検索するか推定するために使用
# total_documents = len(all_splits_resume)
# third = total_documents // 3
# for i, document in enumerate(all_splits_resume):
#     if i < third:
#         document.metadata["section"] = "beginning"
#     elif i < 2 * third:
#         document.metadata["section"] = "middle"
#     else:
#         document.metadata["section"] = "end"


# collectionへデータを追加
collection.add(
    ids=["company", "resume"],
    documents=[
        docs_company,
        docs_resume,
    ])
# vector_store.delete(ids=["1", "2", "3"])

# collection = persistent_client.get_collection(name="collection_name")
# collection = persistent_client.get_collection(name="collection_name")
# print(collection.count())
