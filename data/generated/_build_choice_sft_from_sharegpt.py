import json
import random
import re
from pathlib import Path

SOURCE = Path(r"C:\Users\kanli\Downloads\datasets-lLdQc-f3blWz-sharegpt-2026-11-01.json")
OUTPUT = Path("data/generated/choice-sft-2000-from-sharegpt.jsonl")
RNG = random.Random(20261101)

STOP = set("的了吗呢啊是有在与和及或并且而被把对中上下一些一个这种那其将可以进行作为通过根据关于如何什么哪些怎样是否如果因为所以以及我们你我它他们其中该为从到等" )


def ngrams(text):
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff" and c not in STOP]
    return {"".join(chars[i:i + 2]) for i in range(max(0, len(chars) - 1))}


def clean_sentences(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", ">", "---", "***")):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.、)]\s*", "", line)
        lines.append(line)
    chunks = []
    for line in lines:
        chunks.extend(re.split(r"(?<=[。！？；.!?;])\s*", line))
    out = []
    for item in chunks:
        item = re.sub(r"\*\*|`|\\", "", item).strip(" ：:，,。；;\t")
        if not (20 <= len(item) <= 150):
            continue
        if len(ngrams(item)) < 5:
            continue
        if item not in out:
            out.append(item)
    return out


def main():
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    parsed = []
    for idx, row in enumerate(rows):
        msgs = row.get("messages", [])
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        assistant = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "assistant"), "")
        claims = clean_sentences(assistant)
        if user and claims:
            qgrams = ngrams(user)
            ranked = sorted(claims, key=lambda s: (len(qgrams & ngrams(s)), -abs(len(s) - 65)), reverse=True)
            parsed.append({"idx": idx, "q": user.strip(), "claims": ranked})

    rich = [r for r in parsed if len(r["claims"]) >= 4]
    if len(rich) < 667:
        raise RuntimeError(f"Not enough source rows with four extractable claims: {len(rich)}")

    # Sample independently so all three task types are represented even though
    # the source contains fewer than 2000 conversations.
    singles = RNG.sample(rich, 667)
    multiples = RNG.sample(rich, 667)
    judgements = RNG.sample(rich, 666)

    records = []
    letters = "ABCD"
    for row in singles:
        correct = row["claims"][0]
        distractor_pool = [s for other in parsed if other["idx"] != row["idx"] for s in other["claims"][:3]
                           if len(s) >= 20 and s != correct]
        # Favor same-topic distractors while keeping them sourced from other training rows.
        qg = ngrams(row["q"])
        distractor_pool.sort(key=lambda s: len(qg & ngrams(s)), reverse=True)
        distractors = []
        for s in distractor_pool:
            if s != correct and s not in distractors and abs(len(s) - len(correct)) < 65:
                distractors.append(s)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = [correct] + distractors
        RNG.shuffle(options)
        answer = letters[options.index(correct)]
        prompt = f"{row['q']}\n根据训练集中对应回答的核心结论，以下哪项表述正确？\n" + "\n".join(f"{letters[i]}. {v}" for i, v in enumerate(options)) + "\n只输出正确选项字母。"
        records.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]})

    for row in multiples:
        true_claims = row["claims"][:3]
        distractor = next((s for other in parsed if other["idx"] != row["idx"] for s in other["claims"][:3]
                          if s not in true_claims and abs(len(s) - 60) < 90), None)
        if not distractor:
            continue
        options = true_claims + [distractor]
        RNG.shuffle(options)
        answer = json.dumps(sorted(letters[i] for i, value in enumerate(options) if value in true_claims), ensure_ascii=False)
        prompt = f"{row['q']}\n根据训练集中对应回答，以下哪些表述能得到该回答的明确支持？\n" + "\n".join(f"{letters[i]}. {v}" for i, v in enumerate(options)) + "\n只输出正确选项字母组成的 JSON 数组。"
        records.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]})

    for row in judgements:
        is_true = RNG.random() < 0.5
        if is_true:
            claim = row["claims"][RNG.randrange(min(4, len(row["claims"])))]
        else:
            qg = ngrams(row["q"])
            candidates = [s for other in parsed if other["idx"] != row["idx"] for s in other["claims"][:4]
                          if s not in row["claims"] and abs(len(s) - 55) < 80]
            candidates.sort(key=lambda s: len(qg & ngrams(s)), reverse=True)
            claim = candidates[0]
        prompt = f"{row['q']}\n陈述：{claim}\n该陈述是否得到这条训练集回答的明确支持？只输出 ✅ 或 ❌。"
        records.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "✅" if is_true else "❌"}]})

    if len(records) != 2000:
        raise RuntimeError(f"Produced {len(records)} rows; expected exactly 2000")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records), encoding="utf-8")
    print(f"source_rows={len(rows)} eligible_rows={len(parsed)} rich_rows={len(rich)} output_rows={len(records)}")
    print(f"single={sum('JSON 数组' not in r['messages'][0]['content'] and '训练集回答的明确支持' not in r['messages'][0]['content'] for r in records)} output={OUTPUT.resolve()} bytes={OUTPUT.stat().st_size}")
    for i in (0, 667, 1334):
        print(json.dumps(records[i], ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
