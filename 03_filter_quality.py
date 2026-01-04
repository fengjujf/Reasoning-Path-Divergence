import os
import glob
import re
import pandas as pd
import torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import config 

TENSOR_PARALLEL_SIZE = torch.cuda.device_count()
WORDS_FOR_EVALUATION = 300 

def load_model():
    """加载 vLLM 模型"""
    print(f"--- Loading vLLM Model: {config.MODEL_PATH_INSTRUCT} ---")
    print(f"    GPUs Available: {TENSOR_PARALLEL_SIZE}")
    try:
        llm = LLM(
            model=config.MODEL_PATH_INSTRUCT,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH_INSTRUCT, trust_remote_code=True)
        return llm, tokenizer
    except Exception as e:
        print(f"[Fatal Error] Failed to load vLLM model: {e}")
        exit(1)

def build_evaluation_prompts(answers, tokenizer):
    """构造评估 Prompt"""
    prompts = []
    valid_indices = [] 
    
    for idx, text in enumerate(answers):
        if not isinstance(text, str) or not text.strip():
            continue
            
        snippet = " ".join(text.split()[-WORDS_FOR_EVALUATION:])
        
        prompt_content = f"""You are an expert evaluator. Your task is to assess the quality of the following text snippet, which is the end of a model-generated answer.

An answer is considered LOW QUALITY if it meets ANY of the following criteria:
1.  **No Clear Answer:** It fails to provide a clear, final answer to the implied question.
2.  **Unfinished:** The text appears to be cut off mid-sentence or is clearly incomplete.
3.  **Repetitive Filler:** It contains non-substantive, repetitive filler phrases (e.g., "Wait, wait, wait...", "Hold on, hold on...").

---
Text to evaluate:
"{snippet}"
---

Based on these criteria, is the answer of acceptable quality? Respond ONLY with `//boxed{{Yes}}` for acceptable quality, or `//boxed{{No}}` for low quality. Your response must be in a parsable format.
"""
        messages = [{"role": "user", "content": prompt_content}]
        
      
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False 
            )
        except TypeError:
            chat_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
        prompts.append(chat_prompt)
        valid_indices.append(idx)
        
    return prompts, valid_indices

def run_quality_filter():
    print(f"--- [Step 3] Quality Filtering (Min {config.MIN_HIGH_QUALITY_ANSWERS} Good Answers) ---")
    
    input_files = sorted(glob.glob(os.path.join(config.PROCESSED_DATA_DIR, "length_filtered_data_*.parquet")))
    if not input_files:
        print(f"[Error] No input files found in {config.PROCESSED_DATA_DIR}. Run Step 02 first.")
        return

    llm, tokenizer = load_model()
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    file_counter = 0

    for file_path in tqdm(input_files, desc="Processing Files"):
        try:
            df = pd.read_parquet(file_path)
            
          
            all_answers = []
            map_idx_to_coord = []
            
            ans_cols = [c for c in df.columns if c.startswith('answer_')]
            
            for r_idx, row in df.iterrows():
                for col in ans_cols:
                    val = row[col]
                    all_answers.append(val)
                    map_idx_to_coord.append((r_idx, col))
            
            prompts, valid_indices = build_evaluation_prompts(all_answers, tokenizer)
            
            if not prompts:
                continue

            outputs = llm.generate(prompts, sampling_params)
            
            quality_map = {}
            
            for i, output in enumerate(outputs):
                original_flat_idx = valid_indices[i]
                row_idx, col_name = map_idx_to_coord[original_flat_idx]
                
                generated_text = output.outputs[0].text
                
                match = re.search(r"//?boxed\{(Yes|NO)\}", generated_text, re.IGNORECASE)
                is_high_quality = bool(match and match.group(1).lower() == 'yes')
                
                if row_idx not in quality_map:
                    quality_map[row_idx] = {}
                quality_map[row_idx][col_name] = is_high_quality

            new_records = []
            
            for r_idx, row in df.iterrows():
                row_quality = quality_map.get(r_idx, {})
                good_answers = []
                for col in ans_cols:
                    if row_quality.get(col, False):
                        good_answers.append(row[col])
                if len(good_answers) >= config.MIN_HIGH_QUALITY_ANSWERS:
                    record = {
                        'question': row['question'],
                        'difficulty': row['difficulty']
                    }
                    for k, ans_text in enumerate(good_answers):
                        record[f'answer_{k}'] = ans_text
                    
                    new_records.append(record)

            if new_records:
                
                output_df = pd.DataFrame(new_records)
                save_chunk(output_df, file_counter)
                file_counter += 1
                
        except Exception as e:
            print(f"    [Error] Processing {file_path}: {e}")
            continue

    print("\n--- [Step 3 Complete] ---")

def save_chunk(df, counter):
    filename = f"quality_filtered_data_{counter:05d}.parquet"
    path = os.path.join(config.PROCESSED_DATA_DIR, filename)
    df.to_parquet(path, index=False)
    print(f"    Saved: {path} ({len(df)} records)")

if __name__ == "__main__":
    run_quality_filter()