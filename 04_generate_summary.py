import os
import glob
import json
import pandas as pd
import torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import config  

TENSOR_PARALLEL_SIZE = torch.cuda.device_count()

PROMPT_TEMPLATE = """You are a specialized AI expert in analyzing mathematical solutions. Your task is to first provide a step-by-step analysis of a solution, and then, based on your analysis, generate a final JSON output that is concise, direct, and method-focused.

### REQUIRED OUTPUT STRUCTURE:
Your response MUST have two distinct parts in the following order:

**Part 1: Analysis & Thinking Process**
- Start this section with the heading `### Analysis`.
- Briefly explain your reasoning as you deconstruct the provided solution. This is your "scratchpad".

**Part 2: Final JSON Output**
- After your analysis, provide the final JSON output enclosed in `//boxed{{}}`.
- This part must contain *only* the `//boxed{{...}}` block and nothing else.

### CONTENT RULES FOR THE FINAL JSON:
1.  **Step Count**: The JSON must contain **strictly 3 to 5 logical steps**.
2.  **Output Style**:
    - **Use direct, active verb phrases.** Start each description with a verb (e.g., "Calculate", "Identify", "Apply").
    - **DO NOT use narrative phrasing** like "The author identifies..." or "The solution then calculates...".
3.  **Abstraction Level**:
    - Be abstract about numbers and variables, but **be specific about the methodology**.
    - **BAD (Too Vague):** "Use a formula to get the result."
    - **BAD (Too Concrete):** "Calculate 1/3 + 1/6 = 1/2."
    - **GOOD (Balanced):** "Combine the individual rates to find the total work rate."

### JSON STRUCTURE SPECIFICATION:
- The root object must have one key: `"logical_steps"`.
- The value of `"logical_steps"` must be a list (`[]`) of step objects.
- Each step object (`{{}}`) must contain two keys:
  - `"step_title"`: A short title for the step (e.g., "Step 1: Combine Rates"). Use `null` if not applicable.
  - `"step_description"`: A concise summary of the action, following all rules above.

### EXAMPLE OF THE COMPLETE TWO-PART OUTPUT:
**Input Solution**: "Pipe A fills a tank in 3 hours, so its rate is 1/3 tank/hr. Pipe B fills it in 6 hours, so its rate is 1/6 tank/hr. Together, their rate is 1/3 + 1/6 = 1/2 tank/hr. Therefore, the time to fill the tank together is the reciprocal of the rate, which is 1 / (1/2) = 2 hours."
**Your Required Output**:
### Analysis
The solution addresses a classic work-rate problem.
1.  First, it calculates the individual rate for each pipe.
2.  Second, it sums these rates to get a combined rate.
3.  Finally, it converts the combined rate back into total time.
The logic is broken down into three clear, abstract steps.

//boxed{{
  "logical_steps": [
    {{
      "step_title": "Step 1: Determine Individual Rates",
      "step_description": "Determine the individual work rate of each component based on the time taken."
    }},
    {{
      "step_title": "Step 2: Combine Rates",
      "step_description": "Combine the individual rates to find the total system work rate."
    }},
    {{
      "step_title": "Step 3: Calculate Total Time",
      "step_description": "Calculate the total time by taking the reciprocal of the combined work rate."
    }}
  ]
}}

---

### YOUR TASK:

**Math Problem**:
{question_text}

**Chain-of-Thought Solution to Analyze**:
{answer_cot}

---

### !!! CRITICAL REQUIREMENTS RECAP !!!
Before generating the final JSON, ensure your output strictly adheres to these constraints:
1.  **Strictly 3-5 Steps**: Consolidate or expand if necessary to fit this range.
2.  **Verb-First Phrasing**: Descriptions MUST start with an action verb (e.g., "Calculate...", "Derive..."). DO NOT use "The solution..." or "Step 1...".
3.  **Abstract Methodology**: Describe the *method*, NOT the specific numbers.
4.  **Format**: The final JSON must be valid and enclosed in `//boxed{{...}}`.

"""


def extract_boxed_json(raw_text):
   
    start_marker = '//boxed{'
    try:
        start_pos = raw_text.find(start_marker)
        if start_pos == -1:
            return {"error": "Delimiter //boxed{ not found in the output."}
        
        json_start_pos = start_pos + len(start_marker)
        brace_level = 1
        for i in range(json_start_pos, len(raw_text)):
            char = raw_text[i]
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            
            if brace_level == 0:
                json_str = raw_text[json_start_pos - 1 : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to decode JSON: {e}", "json_string": json_str}
        return {"error": "Could not find the matching closing brace."}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

def load_vllm_model():
    print(f"--- Loading vLLM Model: {config.MODEL_PATH_INSTRUCT} ---")
    try:
        model = LLM(
            model=config.MODEL_PATH_INSTRUCT,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype='bfloat16',
            gpu_memory_utilization=0.9
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH_INSTRUCT, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer
    except Exception as e:
        print(f"[Fatal Error] Failed to load vLLM: {e}")
        exit(1)

def build_chat_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
   
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



def generate_summaries():
    print(f"--- [Step 4] Generating CoT Summaries ---")
    
   
    input_files = sorted(glob.glob(os.path.join(config.PROCESSED_DATA_DIR, "quality_filtered_data_*.parquet")))
    if not input_files:
        print(f"[Error] No input files found in {config.PROCESSED_DATA_DIR}. Run Step 03 first.")
        return
    if config.TEST_LIMIT:
        print(f"[TEST MODE] Limiting input files to 1 file due to TEST_LIMIT={config.TEST_LIMIT}")
        input_files = input_files[:1]
  
    output_dir = config.DIR_STEP4_SUMMARY
    os.makedirs(output_dir, exist_ok=True)
    print(f"    Output Directory: {output_dir}")

 
    model, tokenizer = load_vllm_model()
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2048)

    global_idx = 0 
    
    for file_path in input_files:
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        try:
            df = pd.read_parquet(file_path)
            if config.TEST_LIMIT:
                print(f"    [TEST MODE] Slicing dataframe to first {config.TEST_LIMIT} rows")
                df = df.head(config.TEST_LIMIT)

            
            prompts_buffer = []
            metadata_buffer = [] 
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Prompts"):
               
                output_json_path = os.path.join(output_dir, f"{global_idx}.json")
                if os.path.exists(output_json_path):
                    global_idx += 1
                    continue 
                
                question_text = row["question"]
                
                ans_cols = [c for c in df.columns if c.startswith('answer_')]
                answers_list = [row[c] for c in ans_cols if pd.notna(row[c])]
                
                for ans_id, answer_cot in enumerate(answers_list):
                    final_prompt_content = PROMPT_TEMPLATE.format(
                        question_text=question_text, 
                        answer_cot=answer_cot
                    )
                    chat_prompt = build_chat_prompt(tokenizer, final_prompt_content)
                    
                    prompts_buffer.append(chat_prompt)
                    metadata_buffer.append({
                        "global_idx": global_idx,
                        "answer_id": ans_id,      
                        "original_question": question_text
                    })
                
            
                global_idx += 1

            if not prompts_buffer:
                continue
                
            print(f"    Running inference on {len(prompts_buffer)} prompts...")
            outputs = model.generate(prompts_buffer, sampling_params)
            
            file_results = {} 
            
            for i, output in enumerate(outputs):
                meta = metadata_buffer[i]
                q_idx = meta["global_idx"]
                ans_id = meta["answer_id"]
                
                raw_text = output.outputs[0].text.strip()
                extracted_data = extract_boxed_json(raw_text)
                
             
                summary_obj = {
                    "answer_id": ans_id,
                    "generated_summary": raw_text, 
                    "logical_steps": []
                }
                
                if isinstance(extracted_data, dict) and "logical_steps" in extracted_data:
                    summary_obj["logical_steps"] = extracted_data["logical_steps"]
                
        
                if q_idx not in file_results:
                    file_results[q_idx] = {
                        "original_question": meta["original_question"],
                        "summaries": []
                    }
                file_results[q_idx]["summaries"].append(summary_obj)
            
         
            print(f"    Saving {len(file_results)} JSON files...")
            for q_idx, data in file_results.items():
                save_path = os.path.join(output_dir, f"{q_idx}.json")
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    
        except Exception as e:
            print(f"    [Error] Failed processing file {file_path}: {e}")
           
            continue

    print(f"\n--- [Step 4 Complete] Summaries generated in {output_dir} ---")

if __name__ == "__main__":
    generate_summaries()