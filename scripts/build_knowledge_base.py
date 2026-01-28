import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ================= 配置区域 =================
DATA_SOURCE_PATH = "./knowledge_data" 
DB_PERSIST_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ===========================================

def build_database():
    if not os.path.exists(DATA_SOURCE_PATH):
        print(f"❌ 错误：目录 {DATA_SOURCE_PATH} 不存在。")
        return

    if os.path.exists(DB_PERSIST_PATH):
        print(f"🧹 清理旧数据库: {DB_PERSIST_PATH} ...")
        shutil.rmtree(DB_PERSIST_PATH)

    all_documents = []

    # 1. 加载 TXT 文件
    print("📂 正在加载 .txt 文件...")
    loader_txt = DirectoryLoader(
        DATA_SOURCE_PATH, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
    )
    docs_txt = loader_txt.load()
    all_documents.extend(docs_txt)
    print(f"   - 找到 {len(docs_txt)} 个 txt 文档")

    # 2. 加载 Markdown 文件 (新增逻辑)
    print("📂 正在加载 .md 文件...")
    loader_md = DirectoryLoader(
        DATA_SOURCE_PATH, 
        glob="**/*.md", 
        loader_cls=TextLoader, # TextLoader 完全可以读取 md 的纯文本内容
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
    )
    docs_md = loader_md.load()
    all_documents.extend(docs_md)
    print(f"   - 找到 {len(docs_md)} 个 md 文档")

    if not all_documents:
        print("⚠️ 警告：没有找到任何文档。")
        return

    # 3. 智能切分 (针对 Markdown 优化)
    print("✂️ 正在切分文本...")
    
    # 我们调整了分隔符的优先级：
    # 优先在 Markdown 标题 (##, ###) 处切断，
    # 其次在 换行符+【 (针对之前的课程表) 处切断，
    # 最后才是普通的换行和句号。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n# ",   # <--- 🟢 新增这个！最高优先级
            "\n## ", 
            "\n### ", 
            "\n【", 
            "\n\n", 
            "\n", 
            "。", 
            "！", 
            "？"
        ]
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"✅ 所有文档已切分为 {len(splits)} 个知识块。")

    # 4. 初始化 Embedding
    print(f"🧠 正在加载 Embedding 模型...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

    # 5. 写入数据库
    print("💾 正在构建向量数据库...")
    vector_store = Chroma(
        collection_name="vtuber_knowledge",
        embedding_function=embedding_function,
        persist_directory=DB_PERSIST_PATH
    )
    
    batch_size = 100
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        vector_store.add_documents(documents=batch)
        print(f"   已处理 {min(i+batch_size, len(splits))}/{len(splits)} 个块...")

    print(f"🎉 知识库更新完毕！支持 txt 和 md。")

if __name__ == "__main__":
    build_database()