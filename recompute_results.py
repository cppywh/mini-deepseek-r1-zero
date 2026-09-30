"""复算四模型100题指标并检查summary；输入results目录，输出核验及同题改错统计。"""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent / "results"
records = {}
for name in ("Base", "GRPO", "Distill", "Instruct-Think"):
    result = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    rows = result["details"]
    assert len(rows) == 100 and [r["index"] for r in rows] == list(range(100))
    for row in rows:
        pred, gold = row["predicted"], row["ground_truth"]
        if pred is None:
            expected = False
        else:
            try:
                expected = abs(float(pred) - float(gold)) <= 1e-4
            except (ValueError, TypeError):
                expected = str(pred).strip() == str(gold).strip()
        assert expected == row["correct"], (name, row["index"], "incorrect saved verdict")
    correct = sum(r["correct"] for r in rows)
    hit_limit = sum(r["hit_limit"] for r in rows)
    eligible = [r for r in rows if not r["hit_limit"]]
    assert correct == result["correct"]
    assert hit_limit == result["hit_limit_count"]
    assert len(eligible) == result["eligible_total"]
    assert sum(r["correct"] for r in eligible) == result["eligible_correct"]
    print(name, correct, "/100; excluded", hit_limit)
    records[name] = rows
base, grpo = records["Base"], records["GRPO"]
assert [(r["index"], r["ground_truth"]) for r in base] == [(r["index"], r["ground_truth"]) for r in grpo]
print("wrong_to_right", sum(not b["correct"] and g["correct"] for b, g in zip(base, grpo)))
print("right_to_wrong", sum(b["correct"] and not g["correct"] for b, g in zip(base, grpo)))
print("all checks passed")
