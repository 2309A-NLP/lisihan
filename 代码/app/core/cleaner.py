# app/core/cleaner.py
import re
import unicodedata
import hashlib


class DataCleaner:
    """数据清洗器"""

    def clean_text(self, text: str) -> str:
        """
        清理文本：去除非打印字符、规范化空白、统一格式

        Args:
            text: 原始文本

        Returns:
            str: 清洗后的文本（保留有效内容）
        """

        if not text:
            return ""

        # 1. 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 2. 移除控制字符
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t\r')

        # 3. Unicode 规范化
        text = unicodedata.normalize('NFKC', text)

        # 4. 处理每一行
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # 移除首尾空白
            line = line.strip()
            if line:  # 只保留非空行
                # 合并连续空格
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)

        # 5. 重新组合（保留段落结构需要1个空行的话单独处理）
        cleaned_text = '\n'.join(cleaned_lines)

        # 6. 敏感词过滤
        cleaned_text = self.filter_sensitive(cleaned_text)

        return cleaned_text

    def deduplicate(self, texts: list) -> list:
        """知识库去重（基于文本hash）"""
        if not texts:
            return []

        seen = set()
        unique_texts = []

        for text in texts:
            # 遍历传入的文本列表
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            # 将字符串编码为字节
            # 创建MD5哈希对象
            # 计算哈希值并转为十六进制字符串
            if text_hash not in seen:
                # 检查当前文本的哈希值是否已经在集合中
                seen.add(text_hash)
                # 将当前文本的哈希值添加到集合中
                unique_texts.append(text)
                # 将当前文本添加到结束列表

        return unique_texts
        # 返回去重后的文本列表

    def filter_sensitive(self, text: str) -> str:
        """
        敏感词过滤
        """
        # 合规精简敏感词库（实际可匹配、项目通用）
        sensitive_words = [
            # 色情低俗
            "色情", "嫖娼", "卖淫", "约炮", "裸聊", "涉黄",
            # 赌博诈骗
            "赌博", "网赌", "彩票作弊", "刷单", "返利", "诈骗", "杀猪盘",
            # 违规引流联系方式
            "微信", "vx", "QQ", "qq", "手机号", "微信号", "QQ号",
            # 违禁交易
            "毒品", "枪支", "弹药", "违禁品", "走私",
            # 辱骂地域歧视
            "傻逼", "脑残", "废物", "垃圾", "地域黑", "种族歧视",
            # 违规推广外链
            "私服", "外挂", "破解版", "翻墙", "加速器"
        ]

        for word in sensitive_words:
            text = text.replace(word, '**')

        return text

    def normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        if not text:
            return ""

        # 移除行首行尾空格
        lines = [line.strip() for line in text.split('\n')]
        # 移除空行，但保留段落结构
        text = '\n'.join(line for line in lines if line)

        return text