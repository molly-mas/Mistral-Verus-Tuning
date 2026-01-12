#!/usr/bin/env python3
"""
Convert Verus benchmark data to PyTorch training format (JSONL)
This script processes the vericoding benchmark and creates train.jsonl, valid.jsonl, and test.jsonl
for use with HuggingFace Transformers on Linux with PyTorch
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict
import warnings

# Suppress multiprocessing warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing')

# Configuration
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# Supported formats: "chat" (instruction-tuned models), "completions", and "text"
# "chat" format works best with instruction-tuned models like Mistral on PyTorch/HuggingFace
DATA_FORMAT = "chat"  # Change to "text" or "completions" if needed

def read_verus_file(file_path: Path) -> Dict[str, str]:
    """Read a Verus file and extract components."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the Verus file structure
    # The vericoding benchmark typically has: preamble, spec, code, postamble
    parts = {
        'preamble': '',
        'spec': '',
        'code': '',
        'postamble': ''
    }
    
    # Simple parsing - you may need to adjust based on actual file structure
    if '// PREAMBLE' in content:
        sections = content.split('// PREAMBLE')
        if len(sections) > 1:
            rest = sections[1]
            if '// SPEC' in rest:
                preamble, rest = rest.split('// SPEC', 1)
                parts['preamble'] = preamble.strip()
                if '// CODE' in rest:
                    spec, rest = rest.split('// CODE', 1)
                    parts['spec'] = spec.strip()
                    if '// POSTAMBLE' in rest:
                        code, postamble = rest.split('// POSTAMBLE', 1)
                        parts['code'] = code.strip()
                        parts['postamble'] = postamble.strip()
    else:
        # Fallback: treat entire file as spec+code
        parts['spec'] = content[:len(content)//2]
        parts['code'] = content[len(content)//2:]
    
    return parts

def create_chat_format(description: str, spec: str, code: str, preamble: str = "", helpers: str = "", postamble: str = "") -> Dict:
    """Create a chat-formatted training example for PyTorch/HuggingFace."""
    system_prompt = """You are an expert in formal verification using the Verus prover. 
Given a formal specification in Verus/Rust, generate the correct implementation with proofs."""
    
    # Build user content with description and full context
    user_parts = []
    
    if description:
        user_parts.append(f"Task: {description}\n")
    
    if preamble:
        user_parts.append(f"Preamble:\n{preamble}\n")
    
    if helpers:
        user_parts.append(f"Helper Functions:\n{helpers}\n")
    
    user_parts.append(f"Specification:\n{spec}")
    
    user_content = "\n".join(user_parts)
    
    # The code to be generated (replacing assume(false))
    assistant_content = code
    if postamble:
        assistant_content = f"{code}\n\n{postamble}"
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()}
        ]
    }

def create_text_format(description: str, spec: str, code: str, preamble: str = "", helpers: str = "", postamble: str = "") -> Dict:
    """Create a text-formatted training example for PyTorch."""
    parts = ["### Task: Implement Verus Specification\n"]
    
    if description:
        parts.append(f"Description: {description}\n")
    
    if preamble:
        parts.append(f"### Preamble:\n{preamble}\n")
    
    if helpers:
        parts.append(f"### Helpers:\n{helpers}\n")
    
    parts.append(f"### Specification:\n{spec}\n")
    parts.append(f"### Implementation:\n{code}")
    
    if postamble:
        parts.append(f"\n{postamble}")
    
    text = "\n".join(parts)
    return {"text": text.strip()}

def create_completions_format(description: str, spec: str, code: str, preamble: str = "", helpers: str = "", postamble: str = "") -> Dict:
    """Create a completions-formatted training example for PyTorch."""
    prompt_parts = ["Implement the following Verus specification:\n"]
    
    if description:
        prompt_parts.append(f"{description}\n")
    
    if preamble:
        prompt_parts.append(f"{preamble}\n")
    
    if helpers:
        prompt_parts.append(f"{helpers}\n")
    
    prompt_parts.append(spec)
    
    prompt = "\n".join(prompt_parts)
    
    completion = code
    if postamble:
        completion = f"{code}\n\n{postamble}"
    
    return {
        "prompt": prompt.strip(),
        "completion": completion.strip()
    }

def process_verus_benchmark(
    verus_specs_dir: str = "./external/finetune_benchmarks/vericoded",
    output_dir: str = "./data",
    max_samples: int = None
):
    """
    Process Verus benchmark files and create PyTorch training data.
    
    Args:
        verus_specs_dir: Path to the 'specs' folder from vericoding benchmark
        output_dir: Output directory for JSONL files
        max_samples: Maximum number of samples to process (None for all)
    """
    verus_dir = Path(verus_specs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all Verus files
    verus_files = list(verus_dir.glob("**/*.rs"))
    
    if not verus_files:
        print(f"No .rs files found in cats {verus_specs_dir}")
        return
    
    print(f"Found {len(verus_files)} Verus files")
    
    # Limit samples if specified
    if max_samples:
        verus_files = verus_files[:max_samples]
        print(f"Processing {len(verus_files)} samples")
    
    # Process files
    data = []
    for file_path in verus_files:
        try:
            parts = read_verus_file(file_path)
            
            # Create formatted example based on selected format
            if DATA_FORMAT == "chat":
                example = create_chat_format(
                    parts['spec'], 
                    parts['code'],
                    parts['preamble'],
                    parts['postamble']
                )
            elif DATA_FORMAT == "text":
                example = create_text_format(
                    parts['spec'], 
                    parts['code'],
                    parts['preamble'],
                    parts['postamble']
                )
            elif DATA_FORMAT == "completions":
                example = create_completions_format(
                    parts['spec'], 
                    parts['code'],
                    parts['preamble'],
                    parts['postamble']
                )
            else:
                raise ValueError(f"Unknown format: {DATA_FORMAT}")
            
            data.append(example)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Successfully processed {len(data)} examples")
    
    # Shuffle data
    random.shuffle(data)
    
    # Split data
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)
    
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    print(f"Split: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    
    # Write JSONL files
    def write_jsonl(data: List[Dict], filename: str):
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Wrote {filepath}")
    
    write_jsonl(train_data, "train.jsonl")
    write_jsonl(valid_data, "valid.jsonl")
    write_jsonl(test_data, "test.jsonl")
    
    print("\n✅ PyTorch training data created successfully!")
    print(f"Format used: {DATA_FORMAT}")
    print(f"\nNext steps:")
    print(f"1. Review the generated files in {output_dir}")
    print(f"2. Run PyTorch training with HuggingFace Transformers:")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer")
    print(f"   model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')")
    print(f"   trainer = Trainer(model=model, args=training_args, train_dataset=train_data)")
    print(f"   trainer.train()")
    print(f"\n3. Or use the script runner with:")
    print(f"   accelerate launch train.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --data_dir {output_dir}")

def process_vericoding_jsonl(jsonl_path: str = "verus_tasks.jsonl", output_dir: str = "./data"):
    """
    Process the vericoding tasks JSONL file (NOT verus_issues.jsonl).
    Use verus_tasks.jsonl which contains valid, compilable vericoding tasks.
    
    Args:
        jsonl_path: Path to verus_tasks.jsonl (default: "verus_tasks.jsonl")
        output_dir: Output directory for PyTorch-formatted JSONL files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = []
    skipped = 0
    total_lines = 0
    
    print(f"Processing {jsonl_path}...")
    print("Reading file...")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                
                if line_num % 100 == 0:
                    print(f"  Processed {line_num} lines, extracted {len(data)} examples...", end='\r')
                
                try:
                    item = json.loads(line)
                    
                    # Extract fields with 'vc-' prefix
                    description = item.get('vc-description', '').strip()
                    spec = item.get('vc-spec', '').strip()
                    code = item.get('vc-code', '').strip()
                    preamble = item.get('vc-preamble', '').strip()
                    helpers = item.get('vc-helpers', '').strip()
                    postamble = item.get('vc-postamble', '').strip()
                    
                    # Skip if spec or code is empty
                    if not spec or not code:
                        skipped += 1
                        continue
                    
                    # Skip if code only contains assume(false) - these are incomplete tasks
                    if 'assume(false)' in code and len(code.strip()) < 50:
                        skipped += 1
                        continue
                    
                    # Create formatted example based on selected format
                    if DATA_FORMAT == "chat":
                        example = create_chat_format(
                            description, spec, code, preamble, helpers, postamble
                        )
                    elif DATA_FORMAT == "text":
                        example = create_text_format(
                            description, spec, code, preamble, helpers, postamble
                        )
                    elif DATA_FORMAT == "completions":
                        example = create_completions_format(
                            description, spec, code, preamble, helpers, postamble
                        )
                    else:
                        raise ValueError(f"Unknown format: {DATA_FORMAT}")
                    
                    data.append(example)
                    
                except json.JSONDecodeError as e:
                    print(f"\nWarning: Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"\nWarning: Error processing line {line_num}: {e}")
                    continue
        
        print(f"\n✓ Finished reading {total_lines} lines")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {jsonl_path}")
        return
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
        return
    
    print(f"✓ Successfully processed {len(data)} valid examples")
    print(f"  Skipped {skipped} incomplete or invalid examples")
    
    # Shuffle and split
    random.shuffle(data)
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)
    
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    # Write files
    def write_jsonl(data: List[Dict], filename: str):
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Wrote {filepath}")
    
    write_jsonl(train_data, "train.jsonl")
    write_jsonl(valid_data, "valid.jsonl")
    write_jsonl(test_data, "test.jsonl")
    
    print("\n✅ PyTorch training data created from JSONL!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python verus_mlx_converter.py <verus_specs_dir> [output_dir]")
        print("  python verus_mlx_converter.py --jsonl [jsonl_file] [output_dir]")
        print("\nExample:")
        print("  python verus_mlx_converter.py ./vericoding-benchmark/specs ./data")
        print("  python verus_mlx_converter.py --jsonl ./vericoding-benchmark/jsonl/verus_tasks.jsonl ./data")
        print("\nIMPORTANT: Use verus_tasks.jsonl (not verus_issues.jsonl) for training!")
        sys.exit(1)
    
    if sys.argv[1] == "--jsonl":
        # Default to verus_tasks.jsonl if no specific file provided
        jsonl_path = sys.argv[2] if len(sys.argv) > 2 else "./vericoding-benchmark/jsonl/verus_tasks.jsonl"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "./data"
        
        # Warn if user specified verus_issues.jsonl
        if "issues" in jsonl_path.lower():
            print("⚠️  WARNING: You're using verus_issues.jsonl which contains non-compiling tasks.")
            print("    For training, use verus_tasks.jsonl instead!")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Exiting. Use verus_tasks.jsonl for training.")
                sys.exit(0)
        
        process_vericoding_jsonl(jsonl_path, output_dir)
    else:
        #verus_dir = sys.argv[1]
        verus_dir = "./external/finetune_benchmarks/vericoded"
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data"
        process_verus_benchmark(verus_dir, output_dir)
