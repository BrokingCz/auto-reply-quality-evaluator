"""CLI 入口：对 auto_replies.json 逐条评分，输出结构化结果。

用法：
  python scripts/run_eval.py --cases 1         # 先跑单条验证（case_01）
  python scripts/run_eval.py --cases 1,5,20    # 跑指定多条
  python scripts/run_eval.py                   # 全量 20 条
  python scripts/run_eval.py --debug 1         # 打印 case_01 的原始 LLM 输出
  python scripts/run_eval.py --output output/  # 指定输出目录

API 配置：环境变量 DASHSCOPE_API_KEY（或项目根 .env，不会提交到 Git）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMJudge  # noqa: E402
from src.metrics import (  # noqa: E402
    ACCURACY_FAIL_THRESHOLD,
    compute_total,
    gate_triggered,
)
from src.report import build_report  # noqa: E402

DATA_FILE = ROOT / "task3_auto_replies.json"


def load_env() -> None:
    """从项目根 .env 读取配置（若存在）；环境变量优先。"""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def load_cases(path: Path = DATA_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_case(judge: LLMJudge, case: dict) -> dict:
    """对单条 case 评分，附上下文信息与聚合结果。"""
    result = judge.score(case["user_question"], case["auto_reply"])
    row = {
        "id": case["id"],
        "user_question": case["user_question"],
        "auto_reply": case["auto_reply"],
        "scores": result,
        "has_error": "error" in result,
        "gate_triggered": False,
        "total": None,
        "bucket": "差",
    }
    if not row["has_error"]:
        row["gate_triggered"] = gate_triggered(result)
        row["total"] = compute_total(result)
        acc = result.get("accuracy", {}).get("score")
        accuracy_fail = isinstance(acc, (int, float)) and acc <= ACCURACY_FAIL_THRESHOLD
        if accuracy_fail:
            row["bucket"] = "差（信息错误）"
        elif row["gate_triggered"]:
            row["bucket"] = "⚠️不合格（瞎编）"
        elif row["total"] >= 4:
            row["bucket"] = "好"
        elif row["total"] >= 3:
            row["bucket"] = "中"
        else:
            row["bucket"] = "差"
    return row


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="客服自动回复质量评估器")
    parser.add_argument("--cases", default="all", help="跑哪几条，如 1 / 1,5,20 / all")
    parser.add_argument("--debug", type=int, default=0, help="打印指定 case 的原始 LLM 输出")
    parser.add_argument("--output", default=str(ROOT / "output"), help="输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = LLMJudge(model=args.model, base_url=args.base_url)
    cases = load_cases()

    if args.debug:
        case = next(c for c in cases if c["id"] == f"case_{args.debug:02d}")
        from src.judge import SYSTEM_PROMPT

        print("==== SYSTEM PROMPT ====")
        print(SYSTEM_PROMPT)
        print("\n==== USER MESSAGE ====")
        print(f"用户问题：{case['user_question']}\n自动回复：{case['auto_reply']}")
        print("\n==== 原始 LLM 输出 ====")
        raw = judge._call([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题：{case['user_question']}\n自动回复：{case['auto_reply']}"},
        ])
        print(raw)
        return

    if args.cases == "all":
        indices = list(range(len(cases)))
    else:
        indices = [int(x) for x in args.cases.split(",")]
        indices = [i - 1 for i in indices]

    rows = []
    for i in indices:
        if not (0 <= i < len(cases)):
            print(f"跳过越界索引: {i + 1}")
            continue
        case = cases[i]
        row = score_case(judge, case)
        rows.append(row)
        tag = "ERR" if row["has_error"] else row["bucket"]
        total = "-" if row["total"] is None else f"{row['total']:.2f}"
        print(f"[{row['id']}] 总分={total} 档位={tag}")

    out_file = out_dir / "scores.json"
    out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {len(rows)} 条评分到 {out_file}")

    res = build_report(out_file, out_dir, title=f"样本 {len(rows)} 条 · {args.model}")
    print(f"评估报告已生成: {res['md_path']}")
    om = res["agg"]["overall_mean"]
    print(f"整体均分: {om:.2f}" if om is not None else "整体均分: 无有效样本")
    print(f"档位: {res['agg']['bucket_counts']}")


if __name__ == "__main__":
    main()
