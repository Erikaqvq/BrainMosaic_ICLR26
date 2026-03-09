import argparse
import json
from pathlib import Path

import httpx


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _head_hint_text(head_name, probs, cfg, language):
    if not isinstance(probs, list) or len(probs) == 0:
        return None
    threshold_cfg = cfg["prompt"].get("hint_thresholds", {})
    old_key_alias = {
        "sentence_mode": "te",
        "subjectivity": "oors",
        "semantic_focus": "su",
    }
    threshold = float(
        threshold_cfg.get(
            head_name,
            threshold_cfg.get(old_key_alias.get(head_name, ""), cfg["prompt"].get("hint_threshold", 0.7)),
        )
    )

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    best_prob = float(probs[best_idx])
    if best_prob < threshold:
        return None

    label_map = cfg["prompt"].get("hint_labels", {})
    labels = None
    if isinstance(label_map, dict):
        if language in label_map and isinstance(label_map.get(language), dict):
            labels = label_map[language].get(head_name)
            if labels is None:
                labels = label_map[language].get(old_key_alias.get(head_name, ""))
        elif head_name in label_map:
            labels = label_map.get(head_name)
    if not isinstance(labels, list) or best_idx >= len(labels):
        if language == "zh":
            labels = [f"{head_name}_{i}" for i in range(len(probs))]
        else:
            labels = [f"{head_name}_{i}" for i in range(len(probs))]

    label = str(labels[best_idx]).strip()
    if language == "zh":
        return f"这个句子可能是{label}"
    return f"This sentence may be {label}."


def _build_hints(sentence_mode_probs, subjectivity_probs, semantic_focus_probs, cfg, language):
    hints = []
    for head_name, probs in [
        ("sentence_mode", sentence_mode_probs),
        ("subjectivity", subjectivity_probs),
        ("semantic_focus", semantic_focus_probs),
    ]:
        hint = _head_hint_text(head_name, probs, cfg, language)
        if hint:
            hints.append(hint)
    return hints


def build_prompt(item, cfg):
    topk_words = item.get("topk_words", [])
    sentence_mode_probs = item.get("sentence_mode_probs", item.get("te_probs", []))
    subjectivity_probs = item.get("subjectivity_probs", item.get("oors_probs", []))
    semantic_focus_probs = item.get("semantic_focus_probs", item.get("su_probs", []))

    keywords = []
    for slot_words in topk_words:
        for group in slot_words:
            if isinstance(group, list) and group:
                keywords.append("、".join([str(w).strip() for w in group if str(w).strip()]))
    keywords = [k for k in keywords if k]
    keywords = keywords[: cfg["prompt"].get("max_keywords", 12)]

    language = cfg["prompt"].get("language", "en").lower()
    hints = _build_hints(
        sentence_mode_probs, subjectivity_probs, semantic_focus_probs, cfg, language
    )
    if language == "zh":
        hint_block = "\n".join(hints) if hints else "无可靠全局提示"
        return (
            "你是一个句子重构助手。\n"
            f"请生成恰好 {cfg['generation']['num_candidates']} 条候选句子。\n"
            f"每条句子最多 {cfg['prompt']['max_chars']} 个字符。\n"
            "每行仅输出一句。\n"
            f"关键词：{'；'.join(keywords)}\n"
            f"全局提示：\n{hint_block}"
        )
    hint_block = "\n".join(hints) if hints else "No reliable global hints."
    return (
        "You are a sentence reconstructor.\n"
        f"Generate exactly {cfg['generation']['num_candidates']} candidate sentences.\n"
        f"Max length per sentence: {cfg['prompt']['max_chars']} characters.\n"
        "Output one sentence per line.\n"
        f"Keywords: {'；'.join(keywords)}\n"
        f"Global hints:\n{hint_block}"
    )


def call_llm(prompt, cfg):
    endpoint = cfg["llm"]["endpoint"]
    model = cfg["llm"]["model"]
    api_key = cfg["llm"]["api_key"]
    timeout_sec = cfg["llm"].get("timeout_sec", 60)

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg["generation"].get("temperature", 0.7),
        "max_tokens": cfg["generation"].get("max_tokens", 200),
    }

    with httpx.Client(timeout=timeout_sec) as client:
        res = client.post(endpoint, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
    content = data["choices"][0]["message"]["content"]
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    return lines[: cfg["generation"]["num_candidates"]]


def reconstruct(config_path):
    cfg = load_json(config_path)
    topk_items = load_json(cfg["input"]["topk_json"])
    out = []

    for item in topk_items:
        prompt = build_prompt(item, cfg)
        cands = call_llm(prompt, cfg)
        out.append(
            {
                "sample_index": item.get("sample_index"),
                "gold_sentence": item.get("gold_sentence", ""),
                "prompt": prompt,
                "candidates": cands,
            }
        )

    save_json({"samples": out}, cfg["output"]["reconstruction_json"])
    print(f"Saved: {cfg['output']['reconstruction_json']}")


def main():
    parser = argparse.ArgumentParser("Standalone semantic-guided sentence reconstruction")
    parser.add_argument("--config", required=True, type=str, help="Path to reconstruction config json")
    args = parser.parse_args()
    reconstruct(args.config)


if __name__ == "__main__":
    main()
