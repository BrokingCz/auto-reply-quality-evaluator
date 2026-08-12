"""一致性验证：评估器档位 vs 人工标注等级（data/human_labels.json）。

输出指标：
- 三级(好/中/差)一致率 + Cohen's κ（去除偶然一致的系数）
- 二元(合格/不合格)一致率 + 混淆矩阵
- Spearman 排序相关（评估总分 vs 人工等级序）
- 分歧 case 清单与归因
"""

from __future__ import annotations

import json
from pathlib import Path

LEVELS = ["好", "中", "差"]  # 顺序即等级：好>中>差
LEVEL_NUM = {lvl: i for i, lvl in enumerate(LEVELS)}

# 评估器档位 → 三级等级（⚠️不合格/信息错误 均映射为差）
_BUCKET_TO_LEVEL = {
    "好": "好", "中": "中", "差": "差",
    "⚠️不合格（瞎编）": "差", "差（信息错误）": "差",
}


def _cohen_kappa(observed: list[tuple[str, str]]) -> float:
    """Cohen's κ：两级序（3x3 混淆矩阵）。"""
    n = len(observed)
    if n == 0:
        return 0.0
    matrix = [[0] * 3 for _ in range(3)]
    for a, b in observed:
        matrix[LEVEL_NUM[a]][LEVEL_NUM[b]] += 1
    po = sum(matrix[i][i] for i in range(3)) / n
    row_tot = [sum(matrix[i]) for i in range(3)]
    col_tot = [sum(matrix[j][i] for j in range(3)) for i in range(3)]
    pe = sum((row_tot[i] / n) * (col_tot[i] / n) for i in range(3))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def _spearman(x: list, y: list) -> float:
    """Spearman：平均秩（并列取平均）后求 Pearson。"""
    def ranks(vals: list) -> list:
        n = len(vals)
        idx = sorted(range(n), key=lambda i: (vals[i], i))
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            avg = (i + 1 + j + 1) / 2.0  # 并列段的平均秩
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** 0.5) * (sum((b - my) ** 2 for b in ry) ** 0.5)
    return num / den if den else 0.0


def validate(scores_path: Path, labels_path: Path) -> dict:
    scores = {r["id"]: r for r in json.loads(scores_path.read_text(encoding="utf-8"))}
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    pairs = []  # (our_level, human_level)
    rows = []
    for lab in labels:
        case_id = lab["id"]
        row = scores.get(case_id)
        if row is None or row["has_error"]:
            continue
        our = _BUCKET_TO_LEVEL.get(row["bucket"], "差")
        human = lab["label"]
        pairs.append((our, human))
        rows.append({
            "id": case_id,
            "our": our,
            "human": human,
            "total": row["total"],
            "agree": our == human,
            "human_evidence": lab["evidence"],
            "bucket": row["bucket"],
        })

    # 三级一致率
    n = len(pairs)
    acc3 = sum(1 for a, b in pairs if a == b) / n
    kappa = _cohen_kappa(pairs)

    # 二元一致率（差=不合格）
    bin_pairs = [(_LEVEL_BIN(a), _LEVEL_BIN(b)) for a, b in pairs]
    acc2 = sum(1 for a, b in bin_pairs if a == b) / n
    matrix2 = {"合格/合格": 0, "合格/不合格": 0, "不合格/合格": 0, "不合格/不合格": 0}
    for a, b in bin_pairs:
        matrix2[f"{a}/{b}"] += 1

    # Spearman：评估总分 vs 人工等级序（好=3,中=2,差=1）
    human_num = [LEVEL_NUM[r["human"]] + 1 for r in rows]  # 好=3,中=2,差=1
    totals = [r["total"] for r in rows]
    rho = _spearman(totals, human_num)

    disagree = [r for r in rows if not r["agree"]]
    agree = [r for r in rows if r["agree"]]

    return {
        "n": n,
        "acc3": acc3,
        "kappa": kappa,
        "acc2": acc2,
        "matrix2": matrix2,
        "spearman": rho,
        "rows": rows,
        "agree": agree,
        "disagree": disagree,
        "human_dist": {lvl: sum(1 for r in rows if r["human"] == lvl) for lvl in LEVELS},
        "our_dist": {lvl: sum(1 for r in rows if r["our"] == lvl) for lvl in LEVELS},
    }


def _LEVEL_BIN(level: str) -> str:
    return "不合格" if level == "差" else "合格"


def render_validation_markdown(v: dict) -> str:
    L: list[str] = []
    L.append("## 六、与人工标注的一致性验证\n")
    L.append(f"> 人工标注来源：`task3_human_ref.json` 的 `annotator_notes`，已人工判定为 好/中/差 并附证据（`data/human_labels.json`）。\n")
    L.append("| 指标 | 值 | 解读 |")
    L.append("| :--- | :--- | :--- |")
    L.append(f"| 三级一致率（好/中/差） | **{v['acc3']:.0%}** | {int(v['acc3'] * v['n'])}/{v['n']} 条与人工同判 |")
    L.append(f"| Cohen's κ | **{v['kappa']:.2f}** | 0.41-0.60 中度一致，0.61-0.80 高度一致 |")
    L.append(f"| 二元一致率（合格/不合格） | **{v['acc2']:.0%}** | 合格/合格 {v['matrix2']['合格/合格']} · 合格/不合格 {v['matrix2']['合格/不合格']} · 不合格/合格 {v['matrix2']['不合格/合格']} · 不合格/不合格 {v['matrix2']['不合格/不合格']} |")
    L.append(f"| Spearman 排序相关 | **{v['spearman']:.2f}** | 评估总分与人工等级排序的相关 |")
    L.append("")
    L.append(f"**分布对比**：人工标注 好{v['human_dist']['好']}/中{v['human_dist']['中']}/差{v['human_dist']['差']}；评估器 好{v['our_dist']['好']}/中{v['our_dist']['中']}/差{v['our_dist']['差']}\n")

    if v["disagree"]:
        L.append(f"### 分歧 case（{len(v['disagree'])} 条）\n")
        for r in sorted(v["disagree"], key=lambda r: r["id"]):
            our_num, human_num = LEVEL_NUM[r["our"]], LEVEL_NUM[r["human"]]
            flag = "评估偏宽" if our_num < human_num else "评估偏严"  # 好=0,中=1,差=2；our_num 更小=我们给分更高=偏宽
            L.append(f"- **{r['id']}**：评估器=**{r['our']}**({r['total']:.2f})，人工=**{r['human']}**（{flag}）\n  - 人工证据：{r['human_evidence']}")
    return "\n".join(L)


def run_validation(scores_path: Path, labels_path: Path) -> dict:
    v = validate(scores_path, labels_path)
    md = render_validation_markdown(v)
    (scores_path.parent / "validation.md").write_text(md, encoding="utf-8")
    return v
