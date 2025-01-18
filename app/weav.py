import os
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import weaviate
from weaviate.classes.config import Configure
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

client = weaviate.connect_to_local(
    # compose.ymlで定義したサービス名
    host="weaviate",
    port=8080,
    headers = {
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
    }
)
print(client.is_ready())


# 使用するLLMモデルを定義
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 会社の魅力を読み込んで文字列分割
file_path_company = "jst.pdf"
loader_company = PyPDFLoader(file_path_company)
docs_company = loader_company.load() # pdf読み込み
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=200, add_start_index=True
)
all_splits_company = text_splitter.split_documents(docs_company)

# 埋め込みモデル
embeddings = OpenAIEmbeddings()

# コレクションの生成
# file = client.collections.create(
#     name="ReqruitFile",
#     vectorizer_config=Configure.Vectorizer.text2vec_openai(),
#     generative_config=Configure.Generative.openai()
# )

col = client.collections.get("ReqruitFile")

print(col)

# # PDFファイルのデータをコレクションへ追加
# with col.batch.dynamic() as batch:
#     for doc in all_splits_company:
#         batch.add_object({
#             "text": doc.page_content
#         })

# クエリ
# response = col.query.near_text(
#     query="何年創業？",
#     limit=2
#     grouped_task="Write a tweet with emojis about these facts."
# )
# print(response)

client.close()
