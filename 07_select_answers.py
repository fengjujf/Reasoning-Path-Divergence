import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import config

def load_selected_ids():
    path = os.path.join(config.OUTPUT_DIR, config.FILE_STEP6_SELECTED_Q)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return set(data['selected_ids'])

def greedy_selection(dist_matrix, n_select):
    n_samples = dist_matrix.shape[0]
    if n_samples <= n_select:
        return list(range(n_samples))
        
    selected_indices = []
    remaining_indices = list(range(n_samples))

    avg_dists = np.mean(dist_matrix, axis=1)
    first_idx = np.argmax(avg_dists)
    
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    for _ in range(n_select - 1):
        if not remaining_indices:
            break
            
        sub_matrix = dist_matrix[np.ix_(remaining_indices, selected_indices)]
        
        avg_dist_to_selected = np.mean(sub_matrix, axis=1)
        
        best_candidate_local_idx = np.argmax(avg_dist_to_selected)
        best_candidate_global_idx = remaining_indices[best_candidate_local_idx]
        
        selected_indices.append(best_candidate_global_idx)
        remaining_indices.remove(best_candidate_global_idx)
        
    return selected_indices

def build_final_dataset():
    print(f"--- [Step 7] Constructing Final Dataset (Greedy Selection) ---")
    
    target_ids = load_selected_ids()
    if not target_ids:
        print("[Error] No selected IDs found. Run Step 06 first.")
        return

    matrix_path = os.path.join(config.OUTPUT_DIR, config.FILE_STEP5_MATRIX)
    if not os.path.exists(matrix_path):
        print("[Error] Matrix file not found. Run Step 05 first.")
        return
    print("    Loading distance matrices...")
    precomputed_matrices = np.load(matrix_path)

    parquet_files = sorted(glob.glob(os.path.join(config.PROCESSED_DATA_DIR, "quality_filtered_data_*.parquet")))
    
    final_dataset = []
    global_idx = 0
    processed_count = 0
    
    print(f"    Scanning {len(parquet_files)} parquet files to match {len(target_ids)} questions...")
    
    for p_file in tqdm(parquet_files, desc="Reading Data"):
        df = pd.read_parquet(p_file)
        
        for _, row in df.iterrows():
            current_id = global_idx
            global_idx += 1
            
            if current_id not in target_ids:
                continue
                
            try:
                matrix_key = f"q_{current_id}_matrix"
                ids_key = f"q_{current_id}_ids"
                
                if matrix_key not in precomputed_matrices:
                    print(f"    [Warning] Missing matrix for ID {current_id}, skipping.")
                    continue
                    
                dist_matrix = precomputed_matrices[matrix_key]
                valid_answer_ids = precomputed_matrices[ids_key]
                
                selected_local_indices = greedy_selection(dist_matrix, config.ANSWERS_PER_QUESTION)
                
                final_answer_keys = [f"answer_{valid_answer_ids[i]}" for i in selected_local_indices]
                
                selected_answers_text = []
                for ans_key in final_answer_keys:
                    text = row.get(ans_key)
                    if text and isinstance(text, str):
                        selected_answers_text.append(text)
                
                for ans_text in selected_answers_text:
                    final_dataset.append({
                        "id": current_id,
                        "instruction": row['question'],
                        "input": "",
                        "output": ans_text,
                        "system": "", 
                        "history": [] 
                    })
                
                processed_count += 1
                
            except Exception as e:
                print(f"    [Error] Processing ID {current_id}: {e}")
                continue
    output_json = os.path.join(config.OUTPUT_DIR, config.FILE_STEP7_FINAL)
    print(f"\n    Matched {processed_count}/{len(target_ids)} questions.")
    print(f"    Generated {len(final_dataset)} training samples (QA pairs).")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"    Saved final dataset to: {output_json}")
    print("--- [Step 7 Complete] Pipeline Finished! ---")

if __name__ == "__main__":
    build_final_dataset()