"""Run the project's exact choice scorer inside the pinned Qwen3.5 container."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

from ..data.io import read_jsonl, write_jsonl
from ..evaluation.choice_accuracy import evaluate_choice_records


def main() -> None:
    model_path = "/model"
    suite = Path("/suite/eval.jsonl")
    output = Path("/output")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype="auto", device_map="auto", trust_remote_code=False)
    model.eval()

    class Generator:
        def generate(self, prompt: str) -> str:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            generation = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                        pad_token_id=processor.tokenizer.eos_token_id)
            return processor.decode(generation[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

    started = time.monotonic()
    result = evaluate_choice_records(list(read_jsonl(suite)), Generator())
    predictions = result.pop("predictions")
    write_jsonl(output / "choice_predictions.jsonl", predictions)
    result.update({"predictions_path": "choice_predictions.jsonl",
                   "elapsed_seconds": round(time.monotonic() - started, 3),
                   "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}})
    (output / "choice_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"sample_count": result["sample_count"], "metrics": result["metrics"],
                      "runtime": result["runtime"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
