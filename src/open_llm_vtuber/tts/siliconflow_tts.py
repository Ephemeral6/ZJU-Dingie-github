import re
import requests
from loguru import logger
from .tts_interface import TTSInterface


class SiliconFlowTTS(TTSInterface):
    def __init__(
        self,
        api_url,
        api_key,
        default_model,
        default_voice,
        sample_rate,
        response_format,
        stream,
        speed,
        gain,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.default_model = default_model
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.response_format = response_format
        self.stream = stream
        self.speed = speed
        self.gain = gain

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        # =========== [链接拦截逻辑] ===========
        original_text = text
        
        # 只删除完整的 URL (http/https 开头的)
        # 不要删除域名片段，避免误删太多内容
        text = re.sub(r'https?://\S+', '', text)
        
        # 清理可能残留的空括号
        text = text.replace('()', '').replace('（）', '')
        text = text.replace('[]', '').replace('【】', '')

        # 4. 调试日志 (非常重要，看看到底删干净没)
        if text.strip() != original_text.strip():
            logger.info(f"🔇 [拦截触发] 原始: '{original_text}' -> 最终: '{text}'")
        # ======================================

        # ... 下面的代码不用动 ...
        
        # 如果删完之后没词了，直接返回（防止报错）
        if not text.strip():
            logger.warning("TTS: 内容全是链接，已跳过生成。")
            return ""

        cache_file = self.generate_cache_file_name(
            file_name_no_ext, file_extension=self.response_format
        )
        payload = {
            "input": text, 
            "response_format": self.response_format,
            "sample_rate": self.sample_rate,
            "stream": self.stream,
            "speed": self.speed,
            "gain": self.gain,
            "model": self.default_model,
            "voice": self.default_voice,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # ... (后续 requests 请求代码保持不变) ...
        try:
            if self.api_url is None:
                 # ...
                return ""
            response = requests.request(
                "POST", self.api_url, json=payload, headers=headers
            )
            # ... (保持原样)
            response.raise_for_status()
            with open(cache_file, "wb") as f:
                f.write(response.content)
            return cache_file
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return ""

    def remove_file(self, filepath: str, verbose: bool = True) -> None:
        super().remove_file(filepath, verbose)

    def generate_cache_file_name(self, file_name_no_ext=None, file_extension="wav"):
        return super().generate_cache_file_name(file_name_no_ext, file_extension)
