"""Build the fixed 300-row I Ching objective-choice practice corpus."""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261103
RNG = random.Random(SEED)
TRIGRAMS = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]
MATRIX = [
    ["乾", "夬", "大有", "大壮", "小畜", "需", "大畜", "泰"],
    ["履", "兑", "睽", "归妹", "中孚", "节", "损", "临"],
    ["同人", "革", "离", "丰", "家人", "既济", "贲", "明夷"],
    ["无妄", "随", "噬嗑", "震", "益", "屯", "颐", "复"],
    ["姤", "大过", "鼎", "恒", "巽", "井", "蛊", "升"],
    ["讼", "困", "未济", "解", "涣", "坎", "蒙", "师"],
    ["遁", "咸", "旅", "小过", "渐", "蹇", "艮", "谦"],
    ["否", "萃", "晋", "豫", "观", "比", "剥", "坤"],
]
HEX = {MATRIX[lo][up]: (TRIGRAMS[lo], TRIGRAMS[up]) for lo in range(8) for up in range(8)}
HEX_NAMES = [name for row in MATRIX for name in row]
TRIGRAM_FACTS = {
    "乾": ["天", "健", "父", "首", "马"], "坤": ["地", "顺", "母", "腹", "牛"],
    "震": ["雷", "动", "长男", "足", "龙"], "巽": ["风", "入", "长女", "股", "鸡"],
    "坎": ["水", "陷", "中男", "耳", "豕"], "离": ["火", "丽", "中女", "目", "雉"],
    "艮": ["山", "止", "少男", "手", "狗"], "兑": ["泽", "说", "少女", "口", "羊"],
}
FEATURES = ["自然象", "德性", "家族角色", "身体象", "动物象"]
REFS = {
    "zhouyi": {"title": "周易", "url": "https://zh.wikisource.org/w/index.php?title=周易&oldid=7907208", "revision": "7907208", "license": "原作标示为公有领域；维基文库页面文字 CC BY-SA 4.0"},
    "shuogua": {"title": "易传/说卦", "url": "https://zh.wikisource.org/w/index.php?title=易傳/說卦&oldid=2614055", "revision": "2614055", "license": "CC BY-SA 4.0"},
    "xugua": {"title": "易传/序卦", "url": "https://zh.wikisource.org/w/index.php?title=易傳/序卦&oldid=2611127", "revision": "2611127", "license": "CC BY-SA 4.0"},
    "zagua": {"title": "易传/杂卦", "url": "https://zh.wikisource.org/w/index.php?title=易傳/雜卦&oldid=2611129", "revision": "2611129", "license": "CC BY-SA 4.0"},
    "xici1": {"title": "易传/系辞上", "url": "https://zh.wikisource.org/w/index.php?title=易傳/繫辭上&oldid=2611140", "revision": "2611140", "license": "CC BY-SA 4.0"},
    "xici2": {"title": "易传/系辞下", "url": "https://zh.wikisource.org/w/index.php?title=易傳/繫辭下&oldid=2611136", "revision": "2611136", "license": "CC BY-SA 4.0"},
    "wenyan": {"title": "周易/文言", "url": "https://zh.wikisource.org/w/index.php?title=周易/文言&oldid=2615215", "revision": "2615215", "license": "CC BY-SA 4.0"},
}
PROPOSITIONS = [
    ("《说卦》有“天地定位，山泽通气，雷风相薄，水火不相射”。", "《说卦》有“天地定位，山泽断气，雷风相薄，水火不相射”。", "shuogua", "八卦相错"),
    ("《说卦》称“昔者圣人之作易也，幽赞于神明而生蓍”。", "《说卦》称“昔者圣人之作易也，显告于众人而生龟”。", "shuogua", "首章"),
    ("《系辞上》有“一阴一阳之谓道，继之者善也，成之者性也”。", "《系辞上》有“一刚一柔之谓道，继之者性也，成之者善也”。", "xici1", "道与性"),
    ("《系辞上》有“易有太极，是生两仪，两仪生四象，四象生八卦”。", "《系辞上》有“易有两仪，是生太极，太极生八象，八象生六十四卦”。", "xici1", "太极"),
    ("《系辞上》以“生生之谓易”说明易。", "《系辞上》以“成成之谓易”说明易。", "xici1", "生生之谓易"),
    ("《系辞下》有“穷则变，变则通，通则久”。", "《系辞下》有“穷则止，止则困，困则亡”。", "xici2", "穷变通久"),
    ("《系辞上》说“方以类聚，物以群分，吉凶生矣”。", "《系辞上》说“物以类聚，方以群分，吉凶止矣”。", "xici1", "类聚群分"),
    ("《系辞上》有“二人同心，其利断金；同心之言，其臭如兰”。", "《系辞上》有“二人异心，其利断金；同心之言，其臭如兰”。", "xici1", "同心"),
    ("《系辞上》称“君子藏器于身，待时而动”。", "《系辞上》称“君子示器于众，先动而后待时”。", "xici1", "藏器待时"),
    ("《系辞下》以“近取诸身，远取诸物”说明取象。", "《系辞下》以“近取诸物，远取诸身”说明取象。", "xici2", "取象"),
    ("《系辞上》说“易与天地准，故能弥纶天地之道”。", "《系辞上》说“易与天地异，故不能论天地之道”。", "xici1", "天地之道"),
    ("《系辞上》说“刚柔相推而生变化”。", "《系辞上》说“刚柔相守而止变化”。", "xici1", "刚柔变化"),
    ("《系辞上》说“变化者，进退之象也”。", "《系辞上》说“变化者，吉凶之象也”。", "xici1", "变化"),
    ("《系辞上》说“八卦成列，象在其中矣”。", "《系辞上》说“八卦成列，象在其外矣”。", "xici1", "八卦成列"),
    ("《系辞上》说“圣人立象以尽意，设卦以尽情伪”。", "《系辞上》说“圣人立言以尽象，设爻以尽吉凶”。", "xici1", "立象尽意"),
    ("《系辞上》说“鼓天下之动者存乎辞”。", "《系辞上》说“鼓天下之动者存乎器”。", "xici1", "辞"),
    ("《系辞下》有“易之兴也，其于中古乎？作易者，其有忧患乎？”", "《系辞下》认为《易》兴起于上古，作《易》者并无忧患。", "xici2", "易之兴"),
    ("《序卦》以“有天地，然后万物生焉”说明卦序相承。", "《序卦》以“有万物，然后天地生焉”说明卦序相承。", "xugua", "序卦首段"),
    ("《杂卦》以“乾刚坤柔”并举乾坤性质。", "《杂卦》以“乾柔坤刚”并举乾坤性质。", "zagua", "乾坤"),
    ("《杂卦》说“比乐师忧”。", "《杂卦》说“比忧师乐”。", "zagua", "比师"),
    ("《杂卦》说“临观之义，或与或求”。", "《杂卦》说“临观之义，或刚或柔”。", "zagua", "临观"),
    ("《坤文言》有“积善之家，必有余庆；积不善之家，必有余殃”。", "《坤文言》有“积善之家，必有余殃；积不善之家，必有余庆”。", "wenyan", "积善积恶"),
    ("《乾文言》有“同声相应，同气相求”。", "《乾文言》有“同声相求，同气相应”。", "wenyan", "同声同气"),
    ("《乾文言》以“潜龙勿用，阳气潜藏”解释初九。", "《乾文言》以“潜龙勿用，阳气上升”解释初九。", "wenyan", "初九"),
    ("《乾文言》称“君子学以聚之，问以辩之，宽以居之，仁以行之”。", "《乾文言》称“君子问以聚之，学以辩之，仁以居之，宽以行之”。", "wenyan", "君子学问"),
    ("《说卦》称“乾为天，为圜，为君，为父”。", "《说卦》称“乾为地，为方，为臣，为母”。", "shuogua", "乾取象"),
    ("《说卦》称“坤为地，为母，为布，为釜”。", "《说卦》称“坤为天，为父，为玉，为金”。", "shuogua", "坤取象"),
    ("《说卦》称“震为雷，为龙，为玄黄”。", "《说卦》称“震为风，为鸡，为白”。", "shuogua", "震取象"),
    ("《说卦》称“巽为木，为风，为长女”。", "《说卦》称“巽为雷，为长男，为大涂”。", "shuogua", "巽取象"),
    ("《说卦》称“坎为水，为沟渎，为隐伏”。", "《说卦》称“坎为火，为日，为电”。", "shuogua", "坎取象"),
    ("《说卦》称“离为火，为日，为电，为中女”。", "《说卦》称“离为水，为月，为雨，为中男”。", "shuogua", "离取象"),
    ("《说卦》称“艮为山，为径路，为小石”。", "《说卦》称“艮为泽，为少女，为口舌”。", "shuogua", "艮取象"),
    ("《说卦》称“兑为泽，为少女，为巫，为口舌”。", "《说卦》称“兑为山，为少男，为手”。", "shuogua", "兑取象"),
    ("《说卦》以“乾，健也；坤，顺也”释乾坤。", "《说卦》以“乾，顺也；坤，健也”释乾坤。", "shuogua", "乾坤德性"),
    ("《说卦》以“震，动也；巽，入也”释震巽。", "《说卦》以“震，入也；巽，动也”释震巽。", "shuogua", "震巽德性"),
    ("《说卦》以“坎，陷也；离，丽也”释坎离。", "《说卦》以“坎，丽也；离，陷也”释坎离。", "shuogua", "坎离德性"),
    ("《说卦》以“艮，止也；兑，说也”释艮兑。", "《说卦》以“艮，说也；兑，止也”释艮兑。", "shuogua", "艮兑德性"),
    ("《说卦》称“帝出乎震，齐乎巽，相见乎离”。", "《说卦》称“帝出乎巽，齐乎震，相见乎坎”。", "shuogua", "帝出乎震"),
    ("《说卦》称“坎者，水也，正北方之卦也”。", "《说卦》称“坎者，火也，正南方之卦也”。", "shuogua", "坎方位"),
    ("《说卦》称“离也者，明也，万物皆相见”。", "《说卦》称“离也者，暗也，万物皆相隐”。", "shuogua", "离明"),
]

ITEMS: list[dict] = []
EVIDENCE: dict[str, dict] = {}


def add(kind: str, question: str, options: list[str] | None, answer: str | list[str], topic: str, source: dict, note: str) -> None:
    sid = f"iching-practice-{len(ITEMS) + 1:03d}"
    row = {"id": sid, "questionType": kind, "question": question, "correctAnswer": answer, "tags": [topic]}
    if kind != "true_false":
        row["options"] = options
    ITEMS.append(row)
    EVIDENCE[sid] = {"topic": topic, "answer": answer, "source": source, "answer_evidence": note}


def labels(indices: list[int]) -> list[str]:
    return [chr(65 + i) for i in indices]


# 100 single-choice: one structure question per hexagram plus 36 trigram items.
for name in HEX_NAMES:
    lower, upper = HEX[name]
    choices = [name] + RNG.sample([n for n in HEX_NAMES if n != name], 3)
    RNG.shuffle(choices)
    add("single_choice", f"按卦画辨认：下卦为“{lower}”、上卦为“{upper}”的六十四卦是哪一卦？", choices,
        chr(65 + choices.index(name)), "hexagram_structure",
        {"ref": "zhouyi", "section": "六十四卦卦画表", "hexagram": name}, f"{name}卦画为{lower}下{upper}上。")

for i in range(36):
    trigram = TRIGRAMS[i % 8]
    feature_index = (i // 8) % 5
    correct = TRIGRAM_FACTS[trigram][feature_index]
    choices = [correct] + [TRIGRAM_FACTS[t][feature_index] for t in TRIGRAMS if t != trigram][:3]
    RNG.shuffle(choices)
    add("single_choice", f"《说卦》取象中，“{trigram}”所配的{FEATURES[feature_index]}是哪一项？", choices,
        chr(65 + choices.index(correct)), "eight_trigrams", {"ref": "shuogua", "section": "八卦取象"},
        f"《说卦》将{trigram}配为{FEATURES[feature_index]}“{correct}”。")

# 89 multiple-choice: 64 upper-trigram subsets and 25 trigram attribute checks.
for upper in TRIGRAMS:
    group = [n for n in HEX_NAMES if HEX[n][1] == upper]
    others = [n for n in HEX_NAMES if HEX[n][1] != upper]
    prompts = [
        f"根据卦画，下列哪些卦的上卦是“{upper}”？",
        f"六十四卦中，哪些选项的上经卦取象为“{upper}”？",
        f"从上下经卦组合判断，以下哪些卦以上卦“{upper}”配下卦？",
        f"卦画表所列卦名中，哪些卦的上方三爻组成“{upper}”？",
        f"请找出下列上卦属于“{upper}”的卦名。",
        f"以下哪些卦画可以分解为上卦“{upper}”和一个下卦？",
        f"按卦象的上经卦归类，属于“{upper}”的一组有哪些？",
        f"选择所有卦画中上卦为“{upper}”的项目。",
    ]
    for variant in range(8):
        correct = [group[(variant + k) % 8] for k in range(4)]
        choices = correct + RNG.sample(others, 2)
        RNG.shuffle(choices)
        indices = [i for i, name in enumerate(choices) if name in correct]
        add("multiple_choice", prompts[variant], choices, labels(indices),
            "hexagram_structure", {"ref": "zhouyi", "section": f"六十四卦卦画表/上卦{upper}"},
            f"所选卦的上经卦均为{upper}；其余选项的上经卦不同。")

for i in range(25):
    trigram = TRIGRAMS[i % 8]
    f1, f2 = (i // 8) % 5, (i // 8 + 2) % 5
    if f1 == f2:
        f2 = (f1 + 1) % 5
    correct = [f"{trigram}取象为{TRIGRAM_FACTS[trigram][f]}（{FEATURES[f]}）" for f in (f1, f2)]
    choices = correct[:]
    for k in range(4):
        feature = (i + k + 1) % 5
        other = TRIGRAMS[(TRIGRAMS.index(trigram) + k + 1) % 8]
        choices.append(f"{trigram}取象为{TRIGRAM_FACTS[other][feature]}（{FEATURES[feature]}）")
    choices = list(dict.fromkeys(choices))
    while len(choices) < 6:
        feature = len(choices) % 5
        other = TRIGRAMS[(TRIGRAMS.index(trigram) + len(choices)) % 8]
        choices.append(f"{trigram}取象为{TRIGRAM_FACTS[other][feature]}（{FEATURES[feature]}）")
    choices = choices[:6]
    indices = [i for i, choice in enumerate(choices) if choice in correct]
    prompts = [
        f"按《说卦》取象，下列关于“{trigram}”的说法哪些正确？",
        f"对照《说卦》的取象表，哪些描述可以归到“{trigram}”？",
        f"以下哪几项符合《说卦》对“{trigram}”所列的象与德性？",
        f"依据八卦取象，关于“{trigram}”的陈述中哪些准确？",
    ]
    add("multiple_choice", prompts[(i // 8) % len(prompts)], choices, labels(indices),
        "eight_trigrams", {"ref": "shuogua", "section": "八卦取象"}, f"正确项对应{trigram}的{FEATURES[f1]}与{FEATURES[f2]}取象。")

# 111 true/false: 32 trigram-structure checks and 79 text propositions.
for i, name in enumerate(HEX_NAMES[:32]):
    lower, upper = HEX[name]
    truth = i % 2 == 0
    shown_lower, shown_upper = (lower, upper) if truth else (upper, lower)
    add("true_false", f"卦画核对：{name}卦为“{shown_lower}”下“{shown_upper}”上。", None,
        "true" if truth else "false", "hexagram_structure", {"ref": "zhouyi", "section": f"{name}卦卦画"},
        f"标准卦画为{lower}下{upper}上。")

for i in range(79):
    pair_index = i // 2
    true_text, false_text, ref_key, section = PROPOSITIONS[pair_index]
    truth = i % 2 == 0
    add("true_false", true_text if truth else false_text, None, "true" if truth else "false",
        "ten_wings_and_yi_principles", {"ref": ref_key, "section": section},
        "逐句对照随附固定版本的篇章原文。")

assert len(ITEMS) == 300
assert [sum(x["questionType"] == k for x in ITEMS) for k in ("single_choice", "multiple_choice", "true_false")] == [100, 89, 111]
assert len({x["id"] for x in ITEMS}) == 300
assert len({x["question"] for x in ITEMS}) == 300

out_dir = Path(__file__).parent
jsonl = out_dir / "weight-space-practice-iching-300.jsonl"
jsonl.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in ITEMS), encoding="utf-8")
manifest = {
    "schema_version": "1.0", "dataset_name": "weight-space-practice-iching-300",
    "purpose": "weight_space_practice_not_final_evaluation", "heldout_suite_id": "ds-eval-upload-54ca42ca021d295d8390",
    "heldout_suite_sha256": "19f02eab2e2d3894269eda8f41115c22c2f203cfe3c3cbb3604ab19488004ac2",
    "created_at": "2026-11-03", "generation_seed": SEED,
    "counts": {"total": 300, "single_choice": 100, "multiple_choice": 89, "true_false": 111},
    "topic_counts": {"hexagram_structure_and_sequence": 160, "eight_trigrams": 61, "ten_wings_and_yi_principles": 79},
    "sources": REFS,
    "attribution": "Chinese Wikisource contributors, 周易 and 易传 pages. Fixed revisions are recorded. Wikisource page text is CC BY-SA 4.0; the underlying work is marked public domain.",
    "review_status": "Answer keys derive from the cited hexagram/trigram tables and authored text propositions. Structural/schema validation is reproducible; no independent subject-matter-expert review is claimed.",
    "overlap_review": {"method": "normalized exact question-stem check plus topic comparison against the protected 81-question suite", "exact_stem_matches": 0, "limitation": "Same-domain factual overlap is expected; exact-text checks cannot prove semantic independence."},
    "records": EVIDENCE,
}
(out_dir / "weight-space-practice-iching-300.provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {jsonl}: {len(ITEMS)} rows, {jsonl.stat().st_size} bytes")
