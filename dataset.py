"""Dataset contract, prompts and conservative multiple-choice scoring (stdlib only)."""
import json
import math
import re
from pathlib import Path


def load_dataset(path, video_root):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_records(records, video_root)


def validate_records(records, video_root):
    """Validate in-memory records and group them by resolved video path."""
    if not isinstance(records, list) or not records:
        raise ValueError("标注必须是非空 JSON 数组")
    groups, ids = {}, set()
    for i, source in enumerate(records):
        if not isinstance(source, dict):
            raise ValueError(f"第 {i} 条标注必须为对象")
        q = dict(source)
        for key in ("id", "video", "question"):
            if not isinstance(q.get(key), str) or not q[key].strip():
                raise ValueError(f"第 {i} 条的 {key} 必须为非空字符串")
        if q["id"] in ids:
            raise ValueError(f"重复问题 ID: {q['id']}")
        ids.add(q["id"])
        video = (Path(video_root).expanduser() / q["video"]).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        timestamp = q.get("timestamp")
        if timestamp is not None:
            if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)):
                raise ValueError(f"{q['id']}: timestamp 必须为秒数或 null")
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"{q['id']}: timestamp 必须是有限非负数")
        q["timestamp"] = timestamp
        options = q.get("options", [])
        if not isinstance(options, list) or (options and not 2 <= len(options) <= 26):
            raise ValueError(f"{q['id']}: options 应为空数组或包含 2–26 个选项")
        if any(not isinstance(o, str) or not o.strip() for o in options):
            raise ValueError(f"{q['id']}: 每个选项必须为非空字符串")
        q["options"] = options
        answer = q.get("answer")
        if options and answer is not None:
            if isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer < len(options):
                answer = chr(65 + answer)
            elif isinstance(answer, str) and answer.upper() in [chr(65 + n) for n in range(len(options))]:
                answer = answer.upper()
            else:
                raise ValueError(f"{q['id']}: answer 应为选项字母或从 0 开始的整数索引")
        elif answer is not None and not isinstance(answer, str):
            raise ValueError(f"{q['id']}: 开放式答案必须为字符串")
        q["answer"] = answer
        groups.setdefault(str(video), []).append(q)
    for questions in groups.values():
        questions.sort(key=lambda q: float("inf") if q["timestamp"] is None else q["timestamp"])
    return groups


def build_prompt(q):
    if not q["options"]:
        return f"Answer the question using the video frames.\nQuestion: {q['question']}"
    options = "\n".join(f"{chr(65 + i)}. {s}" for i, s in enumerate(q["options"]))
    return (f"Answer the multiple-choice question using the video frames.\n"
            f"Question: {q['question']}\nOptions:\n{options}\n"
            "Return only the letter of the best option.")


def parse_choice(text, options):
    # Reject prose containing several letters instead of guessing a score.
    match = re.fullmatch(r"\s*(?:answer\s*[:：]\s*)?[\(\[]?([A-Z])[\)\]]?[.。]?\s*", text, re.I)
    if match and 0 <= ord(match[1].upper()) - 65 < len(options):
        return match[1].upper()
    return None


def summarize(results):
    scored = [r for r in results if r.get("correct") is not None]
    return {"answered": len(results), "scored_multiple_choice": len(scored),
            "correct": sum(r["correct"] for r in scored),
            "accuracy": sum(r["correct"] for r in scored) / len(scored) if scored else None}
