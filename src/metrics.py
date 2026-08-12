"""指标体系：五维护 rubric、权重、否决门与聚合规则。

设计依据（详见 README.md / PROCESS.md）：
- 业务方四词（准确 / 有用 / 语气好 / 不瞎编）映射为四个维度；
- human_ref 人工标注揭示的高频失败模式（把责任推回用户、未收齐必要信息）
  映射为"有用""槽位"两个高权重维度；
- "不瞎编"独立为否决门：编造 = 信任 / 法律风险，必须独立拦截，不参与加权，
  否则会被其他高分平均掩盖。
"""

# ---- 加权评分维度（行动力 = 有用 + 槽位，合计 0.8，主导总分）----
# 校准依据（Step 4 诊断）：准确在 18/20 条为 5 分、语气几乎恒定，
# 二者无区分度，仅为"必要项"而非"加分项"；人工标准聚焦"是否主动解决"。
METRIC_WEIGHTS = {
    "helpfulness": 0.5,         # 有用：是否针对具体问题、主动解决
    "slot_completeness": 0.3,   # 槽位：是否收齐解决所需的必要信息（订单号等）
    "accuracy": 0.1,            # 准确：说出的信息是否事实正确（必要项）
    "tone": 0.1,                # 语气：礼貌与情绪共情（必要项）
}

# 准确降档：存在实质错误信息（≤2）→ 强制"差"，避免信息错误被高平均掩盖
ACCURACY_FAIL_THRESHOLD = 2

# ---- 否决门维度 ----
GATE_METRIC = "faithfulness"    # 不瞎编
GATE_THRESHOLD = 2              # ≤2 即触发否决（存在明确编造）

SCORED_METRICS = list(METRIC_WEIGHTS.keys())
ALL_METRICS = SCORED_METRICS + [GATE_METRIC]

# ---- 档位划分（用于决策分级）----
BUCKETS = [(4.0, "好"), (3.0, "中"), (0.0, "差")]


def bucket_of(total: float) -> str:
    """将总分映射为 好 / 中 / 差 档位。"""
    for threshold, label in BUCKETS:
        if total >= threshold:
            return label
    return "差"


def compute_total(scores: dict) -> float:
    """加权总分 = Σ(维度分 × 权重)。分数非法(None)时视为缺失，不参与。"""
    total, used_weight = 0.0, 0.0
    for m, w in METRIC_WEIGHTS.items():
        s = scores.get(m, {}).get("score")
        if isinstance(s, (int, float)) and 1 <= s <= 5:
            total += s * w
            used_weight += w
    if used_weight == 0:
        return 0.0
    return total / used_weight


def gate_triggered(scores: dict) -> bool:
    """否决门：faithfulness ≤ GATE_THRESHOLD 即不合格。"""
    s = scores.get(GATE_METRIC, {}).get("score")
    return isinstance(s, (int, float)) and s <= GATE_THRESHOLD


# ---- rubric 锚点（LLM 评分判据 + 报告展示共用同一事实源）----
RUBRIC = {
    "helpfulness": {
        "question": "有用：是否针对用户的具体问题、主动解决，而非把动作推回给用户",
        "anchors": {
            5: "主动承接解决：给出具体信息/承诺查单/明确可执行路径，针对用户具体情况",
            4: "有明确可执行路径，但更主动（如先查单/先追问）会更好",
            3: "有部分可参考信息，但主要依赖用户自行操作",
            2: "通用模板，把动作推回用户（『建议您自行…联系…』）",
            1: "答非所问，或完全无法解决用户问题",
        },
    },
    "slot_completeness": {
        "question": "槽位：是否识别解决该问题所需的必要信息并在缺失时主动追问——硬信息（订单号/账号/商品/尺码等）与澄清类信息（用户卡在哪个环节、想要哪种方案、具体是哪两件商品）都算",
        "needs_slot_hint": "先判断该问题是否必须额外信息才能给出针对性解决（包括澄清用户的具体困难/偏好）；若确实无需，needs_slot=false，槽位直接给 5 分；若需要澄清却没问，评分应 ≤3",
        "anchors": {
            5: "正确识别必要槽位，且缺失时主动追问，为后续解决铺路",
            4: "识别主要槽位，但追问不完整（如只问了一部分）",
            3: "未明确追问，但给出的信息足以继续推进",
            2: "未识别/未追问必要槽位，直接甩通用流程",
            1: "需要槽位才能解决却完全忽略，无法推进",
        },
    },
    "accuracy": {
        "question": "准确：说出的信息是否事实/政策正确（只评『说出来的对不对』）",
        "anchors": {
            5: "所有事实/政策/参数均正确",
            4: "主要正确，个别措辞不精确或无碍的细节偏差",
            3: "存在一处明显的事实偏差或含糊表述",
            2: "存在多处事实错误",
            1: "关键事实/政策错误（可能误导用户）",
        },
    },
    "tone": {
        "question": "语气：礼貌程度与情绪共情是否匹配（用户不满/害怕/焦急时安抚尤为重要）",
        "anchors": {
            5: "礼貌 + 主动安抚用户情绪",
            4: "礼貌得体，情绪抚慰一般",
            3: "基本礼貌，个别表述生硬",
            2: "语气冷漠或敷衍",
            1: "冒犯用户",
        },
    },
    "faithfulness": {
        "question": "不瞎编：有无编造具体事实/政策/赔偿数字（否决维度，≤2 判不合格）",
        "anchors": {
            5: "全部信息真实、可溯源",
            4: "个别惯例性说法（如『一般 1-3 个工作日』），无实质风险",
            3: "有缺乏依据的具体断言，但接近合理",
            2: "明确编造具体事实/政策数字",
            1: "严重编造（虚构政策/承诺）",
        },
    },
}


def rubric_for_prompt() -> str:
    """生成嵌入 judge prompt 的评分手册文本。"""
    lines = []
    for m in ["helpfulness", "slot_completeness", "accuracy", "tone", "faithfulness"]:
        r = RUBRIC[m]
        lines.append(f"{m}: {r['question']}")
        if "needs_slot_hint" in r:
            lines.append(f"   (注意) {r['needs_slot_hint']}")
        for score in sorted(r["anchors"], reverse=True):
            lines.append(f"   {score}= {r['anchors'][score]}")
    return "\n".join(lines)
