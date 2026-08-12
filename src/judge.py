"""LLMJudge：基于 Qwen（DashScope OpenAI 兼容接口）的评分器。

职责：
1. 拼 prompt：用户问题 + 自动回复 + 五维护 rubric → 指令 Qwen 严格返回 JSON；
2. 调用 API（requests），网络错误自动重试（指数退避）；
3. 解析并校验 JSON；格式非法时带修复指令重试；
4. 失败则明确标记，绝不伪造分数。
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

from .metrics import rubric_for_prompt

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"

SYSTEM_PROMPT = f"""你是一名资深电商客服质量评审员。你会收到一段「用户问题」和一段「系统自动回复」。请严格按下面的 5 个维度对自动回复评分，每个维度给出整数分（1-5）和简短中文理由（不超过 30 字）。

【评分手册】
{rubric_for_prompt()}

【评分原则】
- 只评判自动回复本身的质量，不要假设任何标准答案或参考回复；
- 优先判断「是否解决了用户的实际问题」，而非信息是否面面俱到；
- 严格对照锚点评分，同一档位的回复给相同分数，不要手松或手紧；
- 不要因为回复里写了「请联系客服」这类兜底话术而降低有用性——以「是否主动承接解决」为准；
- 「让用户自己查」是减分项：若回复把关键动作（查询/核实/联系）全部推给用户自行完成，helpfulness 应 ≤3；若已直接回答核心问题、但把更深入的信息推给用户自查（如『请自行查看详情页』），helpfulness 同样应 ≤3。

【输出要求】
只输出一个 JSON 对象，不要输出任何其他文字、注释或 Markdown 代码块标记：
{{
  "helpfulness": {{"score": 1, "reason": "..."}},
  "slot_completeness": {{"score": 1, "reason": "...", "needs_slot": true}},
  "accuracy": {{"score": 1, "reason": "..."}},
  "tone": {{"score": 1, "reason": "..."}},
  "faithfulness": {{"score": 1, "reason": "..."}}
}}"""

_REPAIR_NOTE = "上次输出无法解析为合法 JSON。请重新输出，且只输出一个合法的 JSON 对象，不要包含任何额外文字。"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMJudge:
    """OpenAI 兼容接口的评分器（默认 DashScope/Qwen，可换任意兼容端点）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 60,
        max_attempts: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts

        if not self.api_key:
            raise ValueError(
                "缺少 API key：请设置环境变量 DASHSCOPE_API_KEY，或在 .env 中配置。"
            )

    # ---- 网络层 ----
    def _call(self, messages: list[dict]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:  # 网络抖动/超时
                last_err = e
            time.sleep(2 * (attempt + 1))  # 指数退避：2s, 4s
        raise RuntimeError(f"API 调用失败：{last_err}")

    # ---- 解析层 ----
    @staticmethod
    def _parse_json(raw: str) -> dict:
        """从 LLM 输出中提取 JSON 并校验字段完整性。"""
        m = _JSON_RE.search(raw)
        if not m:
            raise ValueError("输出中未找到 JSON")
        obj = json.loads(m.group(0))

        required = {
            "helpfulness": ("score", "reason"),
            "slot_completeness": ("score", "reason", "needs_slot"),
            "accuracy": ("score", "reason"),
            "tone": ("score", "reason"),
            "faithfulness": ("score", "reason"),
        }
        for dim, fields in required.items():
            if dim not in obj or not isinstance(obj[dim], dict):
                raise ValueError(f"缺少维度: {dim}")
            for f in fields:
                if f not in obj[dim]:
                    raise ValueError(f"维度 {dim} 缺少字段: {f}")
            score = obj[dim]["score"]
            if not isinstance(score, int) or not (1 <= score <= 5):
                raise ValueError(f"维度 {dim} 分数非法: {score}")

        ns = obj["slot_completeness"].get("needs_slot")
        if not isinstance(ns, bool):
            raise ValueError(f"needs_slot 必须为布尔值: {ns}")
        return obj

    # ---- 评分入口 ----
    def score(self, question: str, reply: str) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user = f"用户问题：{question}\n自动回复：{reply}"
        result: dict | None = None
        last_err: Exception | None = None

        for attempt in range(self.max_attempts):
            if attempt > 0:
                user += "\n\n" + _REPAIR_NOTE
            messages = messages[:1] + [{"role": "user", "content": user}]
            try:
                raw = self._call(messages)
                result = self._parse_json(raw)
                break
            except Exception as e:  # 解析失败或调用失败
                last_err = e

        if result is None:
            return {
                "error": f"评分失败（{self.max_attempts} 次尝试后）: {last_err}",
                "helpfulness": {"score": None, "reason": "评分失败"},
                "slot_completeness": {"score": None, "reason": "评分失败", "needs_slot": None},
                "accuracy": {"score": None, "reason": "评分失败"},
                "tone": {"score": None, "reason": "评分失败"},
                "faithfulness": {"score": None, "reason": "评分失败"},
            }
        return result
