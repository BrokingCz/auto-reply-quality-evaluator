"""生成最终评估报告：评分聚合报告 + 一致性验证 合并为一份 eval_report.md。

用法：python scripts/generate_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report import build_report  # noqa: E402
from src.validate import run_validation  # noqa: E402


def main() -> None:
    scores = ROOT / "output" / "scores.json"
    labels = ROOT / "data" / "human_labels.json"
    out = ROOT / "output"

    # 1. 聚合报告（一~五章）
    build_report(scores, out, title="正式评估 · qwen-plus · 校准后")

    # 2. 一致性验证（第六章）
    v = run_validation(scores, labels)

    # 3. 合并：验证章节追加到报告末尾
    report_path = out / "eval_report.md"
    md = report_path.read_text(encoding="utf-8")
    if "## 六、" not in md:
        md += "\n" + (out / "validation.md").read_text(encoding="utf-8")
        report_path.write_text(md, encoding="utf-8")

    print(f"最终报告: {report_path}")
    print(f"整体均分: {v['acc3']:.0%} 一致率, κ={v['kappa']:.2f}")


if __name__ == "__main__":
    main()
