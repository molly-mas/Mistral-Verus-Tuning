import json
from pathlib import Path

INPUT_FILE = Path("external/finetune_benchmarks/jsonl/verus_tasks.jsonl")
OUTPUT_FILE = Path("data/training/benchmark_training_clean.jsonl")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def convert(example):
    instruction = "decide what this should say later"
    input_text = example[""]
