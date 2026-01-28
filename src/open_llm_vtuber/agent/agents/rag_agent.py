from typing import List, Dict, Any, Optional
from loguru import logger
from .basic_memory_agent import BasicMemoryAgent
from ..input_types import BatchInput
import datetime
# 引入 RAG 依赖
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class RAGAgent(BasicMemoryAgent):
    """
    带有 RAG (检索增强生成) 功能的 Agent。
    继承自 BasicMemoryAgent，在构建 Prompt 阶段自动检索向量库并注入上下文。
    """

    def __init__(self, vector_db_path: str, *args, **kwargs):
        # 1. 初始化父类 (BasicMemoryAgent)
        super().__init__(*args, **kwargs)
        
        # 2. 初始化向量检索组件
        logger.info(f"正在加载 RAG 向量数据库: {vector_db_path}")
        try:
            # 必须和 build_knowledge_base.py 使用相同的模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'} # 如果有显卡改成 cuda
            )
            # 加载现有的数据库
            self.vector_store = Chroma(
                persist_directory=vector_db_path, 
                embedding_function=self.embeddings,
                collection_name="vtuber_knowledge" # 必须和构建时一致
            )
            logger.info("RAG 数据库加载成功！")
        except Exception as e:
            logger.error(f"RAG 数据库加载失败，Agent 将退化为普通模式: {e}")
            self.vector_store = None
    
    def _retrieve_context(self, query: str, k: int = 3) -> str:
        """根据用户输入检索相关知识片段"""
        if not self.vector_store:
            return ""
            
        try:
            # 检索相似度最高的 k 个片段
            docs = self.vector_store.similarity_search(query, k=k)
            if not docs:
                return ""
            
            # 拼接片段内容
            context_list = [f"资料{i+1}: {doc.page_content}" for i, doc in enumerate(docs)]
            return "\n\n".join(context_list)
        except Exception as e:
            logger.error(f"检索出错: {e}")
            return ""

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """
        重写父类的消息构建方法。
        在返回给 LLM 之前，拦截用户消息并注入上下文和当前时间信息。
        """
        import datetime  # 建议在文件头部导入，或者在这里局部导入

        # 1. 先让父类干苦力，生成标准的消息列表
        messages = super()._to_messages(input_data)
        
        # 2. 提取用户当前的文本输入 (用于检索)
        user_text = self._to_text_prompt(input_data)

        # 3. 获取当前日期信息 (关键步骤：为课程表查询提供时间基准)
        # 格式示例：2026年03月02日 周一
        now = datetime.datetime.now()
        week_days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        date_info = f"{now.strftime('%Y年%m月%d日')} {week_days[now.weekday()]}"
        
        # 4. 如果有文本，就开始检索
        if user_text and self.vector_store:
            logger.info(f"🔎 RAG 正在检索: {user_text[:20]}...")
            context = self._retrieve_context(user_text)
            
            if context:
                logger.info(f"✅ 检索到 {len(context)} 字符的上下文")
                
                # 5. 找到消息列表中的最后一条 (也就是用户刚刚说的话)
                # messages 结构通常是: [...历史对话..., {role: user, content: ...}]
                if messages and messages[-1]['role'] == 'user':
                    original_content = messages[-1]['content']
                    
                    # 构造新的 Prompt：包含系统时间 + 上下文 + 用户原始问题
                    # 加入【系统时间】是为了让 LLM 能推算出“明天”、“下周”具体是哪一天，从而匹配课程表
                    augmented_content = (
                        f"【系统时间】\n现在是：{date_info}\n\n"
                        f"【参考资料】\n{context}\n\n"
                        f"【用户问题】\n{user_text}\n\n"
                        f"请优先基于上述参考资料回答用户问题。如果资料不足，再使用你的通用知识。"
                        f"对于涉及相对日期的问题（如“明天”），请务必根据【系统时间】进行推算。"
                    )
                    
                    # 替换掉原有的内容
                    # 注意：messages[-1]['content'] 可能是字符串列表(多模态)或字符串
                    if isinstance(original_content, str):
                        messages[-1]['content'] = augmented_content
                    elif isinstance(original_content, list):
                        # 如果是多模态(有图片)，找到 text 部分修改
                        for item in original_content:
                            if item.get('type') == 'text':
                                item['text'] = augmented_content
                                break
        
        return messages