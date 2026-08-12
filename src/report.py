"""报告生成器：将 scores.json 聚合为 Markdown 评估报告与 CSV。

职责：
- 整体得分、各指标分布（均值/标准差）、档位（好/中/差）分布
- 否决门触发情况
- 最差 N 条 case 及根因分析（引用评分 reason）
- 指标相关性（两两 Pearson，观察共现失败模式）
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .metrics import ALL_METRICS, GATE_METRIC, METRIC_WEIGHTS, SCORED_METRICS

METRIC_NAMES_CN = {
    "helpfulness": "有用 Helpfulness",
    "slot_completeness": "槽位 Slot",
    "accuracy": "准确 Accuracy",
    "tone": "语气 Tone",
    "faithfulness": "不瞎编 Faithfulness",
}


def _score_or_none(row: dict, metric: str):
    return row["scores"].get(metric, {}).get("score")


def _pearson(xs: list, ys: list) -> float:
    """两列数值的 Pearson 相关系数（任一含 None 或方差为 0 返回 0）。"""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return 0.0
    xs, ys = zip(*pairs)
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) ** 0.5) * (sum((y - my) ** 2 for y in ys) ** 0.5)
    if den == 0:
        return 0.0
    return num / den


def aggregate(rows: list[dict]) -> dict:
    """对评分结果做全量聚合统计。"""
    valid = [r for r in rows if not r["has_error"]]
    failed = [r for r in rows if r["has_error"]]
    totals = [r["total"] for r in valid if r["total"] is not None]
    gated = [r for r in valid if r["gate_triggered"]]

    metric_stats = {}
    for m in ALL_METRICS:
        vals = [_score_or_none(r, m) for r in valid]
        vals = [v for v in vals if v is not None]
        metric_stats[m] = {
            "mean": statistics.fmean(vals) if vals else None,
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "hist": {s: vals.count(s) for s in range(1, 6)},
        }

    bucket_counts = {"好": 0, "中": 0, "差": 0}
    for r in valid:
        if r["gate_triggered"]:
            bucket_counts.setdefault("不合格（瞎编）", 0)
            bucket_counts["不合格（瞎编）"] = bucket_counts.get("不合格（瞎编）", 0) + 1
        else:
            bucket_counts[r["bucket"]] += 1

    corr = {}
    for i, m1 in enumerate(SCORED_METRICS):
        for m2 in SCORED_METRICS[i + 1:]:
            corr[f"{m1}↔{m2}"] = round(
                _pearson(
                    [_score_or_none(r, m1) for r in valid],
                    [_score_or_none(r, m2) for r in valid],
                ),
                3,
            )

    ranked = sorted(valid, key=lambda r: (r["total"] if r["total"] is not None else -1))
    return {
        "valid": valid,
        "failed": failed,
        "gated": gated,
        "n": len(rows),
        "n_valid": len(valid),
        "n_failed": len(failed),
        "n_gated": len(gated),
        "overall_mean": statistics.fmean(totals) if totals else None,
        "overall_stdev": statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "metric_stats": metric_stats,
        "bucket_counts": bucket_counts,
        "corr": corr,
        "best": ranked[-3:][::-1] if ranked else [],
        "worst": ranked[:3] if ranked else [],
    }


def _hist_bar(hist: dict, width: int = 20) -> str:
    total = sum(hist.values())
    if total == 0:
        return ""
    return "  ".join(
        f"{s}分:{'█' * max(1, round(c / total * width)) if c else '·'}({c})"
        for s, c in sorted(hist.items())
    )


def render_markdown(agg: dict, rows: list[dict], report_title: str) -> str:
    L: list[str] = []
    a = agg
    L.append(f"# 客服自动回复质量评估报告\n")
    L.append(f"> {report_title}\n")

    # 1. 总览
    L.append("## 一、总体结果\n")
    L.append(f"| 指标 | 值 |")
    L.append(f"| :--- | :--- |")
    L.append(f"| 评估样本 | {a['n']} 条（有效 {a['n_valid']}，失败 {a['n_failed']}） |")
    L.append(f"| **整体均分（满分5）** | **{a['overall_mean']:.2f}**（标准差 {a['overall_stdev']:.2f}） |" if a["overall_mean"] is not None else "| 整体均分 | 无有效样本 |")
    L.append(f"| 否决门（瞎编）触发 | **{a['n_gated']}** 条 |")
    b = a["bucket_counts"]
    L.append(f"| 档位分布 | 好 {b.get('好', 0)} / 中 {b.get('中', 0)} / 差 {b.get('差', 0)}"
             + (f" / ⚠️不合格 {b.get('不合格（瞎编）', 0)}" if a['n_gated'] else "") + " |")
    L.append("")

    # 2. 指标分布
    L.append("## 二、各指标分布\n")
    L.append("| 指标 | 权重 | 均值 | 标准差 | 分布 |")
    L.append("| :--- | :-: | :-: | :-: | :--- |")
    for m in ALL_METRICS:
        st = a["metric_stats"][m]
        w = f"{METRIC_WEIGHTS[m]:.2f}" if m in METRIC_WEIGHTS else "否决门"
        L.append(
            f"| {METRIC_NAMES_CN[m]} | {w} | {st['mean']:.2f} | {st['stdev']:.2f} | {_hist_bar(st['hist'])} |"
        )
    L.append("")

    # 3. 指标相关性
    L.append("## 三、指标相关性（Pearson）\n")
    for k, v in a["corr"].items():
        flag = "（强相关）" if abs(v) >= 0.6 else ""
        L.append(f"- `{k}` = **{v}** {flag}")
    L.append("")

    # 4. 最差 3 条
    L.append("## 四、最差 3 条 case 及根因分析\n")
    for i, r in enumerate(a["worst"], 1):
        L.append(f"### {i}. {r['id']} — 总分 {r['total']:.2f}（档位：{r['bucket']}）\n")
        L.append(f"**用户问题**：{r['user_question']}")
        L.append(f"\n**自动回复**：{r['auto_reply']}\n")
        L.append(f"**根因分析**：")
        for m in SCORED_METRICS + ([GATE_METRIC] if r["gate_triggered"] else []):
            v = r["scores"].get(m, {})
            L.append(f"- **{METRIC_NAMES_CN[m]}** {v.get('score')} 分：{v.get('reason', '')}")
        if r["gate_triggered"]:
            L.append(f"\n> ⚠️ 该条触发否决门（不瞎编 ≤ 2），判定不合格。")
        L.append("")

    # 5. 全量明细
    L.append("## 五、全量评分明细\n")
    L.append("| case | 有用 | 槽位 | 准确 | 语气 | 不瞎编 | 总分 | 档位 |")
    L.append("| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |")
    for r in sorted(a["valid"], key=lambda r: r["id"]):
        g = _score_or_none(r, "helpfulness")
        s = _score_or_none(r, "slot_completeness")
        ac = _score_or_none(r, "accuracy")
        t = _score_or_none(r, "tone")
        f = _score_or_none(r, "faithfulness")
        total = f"{r['total']:.2f}" if r["total"] is not None else "-"
        L.append(f"| {r['id']} | {g} | {s} | {ac} | {t} | {f} | {total} | {r['bucket']} |")
    if a["failed"]:
        L.append("\n**评分失败条目**：" + ", ".join(r["id"] for r in a["failed"]))
    L.append("")

    return "\n".join(L)


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "helpfulness", "slot", "accuracy", "tone", "faithfulness", "total", "bucket"])
        for r in rows:
            writer.writerow([
                r["id"],
                _score_or_none(r, "helpfulness"),
                _score_or_none(r, "slot_completeness"),
                _score_or_none(r, "accuracy"),
                _score_or_none(r, "tone"),
                _score_or_none(r, "faithfulness"),
                round(r["total"], 2) if r["total"] is not None else "",
                r["bucket"],
            ])


def build_report(scores_path: Path, out_dir: Path, title: str) -> dict:
    rows = json.loads(scores_path.read_text(encoding="utf-8"))
    agg = aggregate(rows)
    md = render_markdown(agg, rows, title)
    md_path = out_dir / "eval_report.md"
    md_path.write_text(md, encoding="utf-8")
    csv_path = out_dir / "scores.csv"
    write_csv(rows, csv_path)
    return {"agg": agg, "md_path": md_path, "csv_path": csv_path}
