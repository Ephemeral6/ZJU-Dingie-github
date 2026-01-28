import abc
import os
import asyncio
import re
from loguru import logger

class TTSInterface(metaclass=abc.ABCMeta):
    
    def _filter_stream_text(self, text: str) -> str:
        """
        【状态机过滤器】
        解决流式传输（Streaming）造成的句子截断问题。
        它维护一个 self._in_brackets 状态，跨数据包记忆是否处于"括号静音区"。
        """
        if not text:
            return text
            
        # === 1. 初始化状态 (Lazy Initialization) ===
        # 防止子类没有调用 super().__init__() 导致变量不存在
        if not hasattr(self, "_in_brackets"):
            self._in_brackets = False
            
        filtered_chars = []
        
        # === 2. 逐字扫描 (State Machine) ===
        for char in text:
            # 检测到左括号 -> 开启静音状态
            if char in ('(', '（'):
                self._in_brackets = True
                continue # 跳过括号本身，不读
            
            # 检测到右括号 -> 关闭静音状态
            if char in (')', '）'):
                self._in_brackets = False
                continue # 跳过括号本身，不读
            
            # 只有当"不在括号里"时，才收集这个字符
            if not self._in_brackets:
                filtered_chars.append(char)
        
        # 将收集到的字符重新拼成字符串
        result = "".join(filtered_chars)
        
        # === 3. 兜底清洗 ===
        # 为了以防万一（比如 LLM 没加括号，且链接恰好在一个包里完整出现），
        # 我们还是保留这个正则作为第二道防线。
        result = re.sub(r'https?://\S+', '', result)
        
        # === 调试日志 ===
        # 如果发生了过滤行为，打印日志方便观察
        if len(result) != len(text):
            # logger.debug(f"🔇 [流式静音] 状态:{self._in_brackets} | 原文片段:{text[:10]}... -> 清洗后:{result[:10]}...")
            pass
            
        return result

    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Asynchronously generate speech audio file using TTS.
        By default, this runs the synchronous generate_audio in a coroutine.
        """
        # 【关键修改】进入 TTS 生成前，先通过状态机清洗
        safe_text = self._filter_stream_text(text)

        # 如果清洗后没词了（比如全是链接），直接返回空或跳过，
        # 但为了防止下游报错，我们还是传进去，让具体引擎自己处理空字符串。
        return await asyncio.to_thread(self.generate_audio, safe_text, file_name_no_ext)

    @abc.abstractmethod
    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext (optional and deprecated): str
            name of the file without file extension

        Returns:
        str: the path to the generated audio file
        """
        raise NotImplementedError

    def remove_file(self, filepath: str, verbose: bool = True) -> None:
        """
        Remove a file from the file system.
        """
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} does not exist")
            return
        try:
            logger.debug(f"Removing file {filepath}") if verbose else None
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Failed to remove file {filepath}: {e}")

    def generate_cache_file_name(self, file_name_no_ext=None, file_extension="wav"):
        """
        Generate a cross-platform cache file name.
        """
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if file_name_no_ext is None:
            file_name_no_ext = "temp"

        file_name = f"{file_name_no_ext}.{file_extension}"
        return os.path.join(cache_dir, file_name)
