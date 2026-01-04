import os
import numpy as np
import json
import pandas as pd
from tqdm.auto import tqdm
import config  

def calculate_diversity_metric(dist_matrix):
    n_samples = dist_matrix.shape[0]
    if n_samples < 2: 
        return 0.0

    mean_distances_per_point = np.sum(dist_matrix, axis=1) / (n_samples - 1)
    
    return float(np.max(mean_distances_per_point))

def select_top_problems():
    print(f"--- [Step 6] Selecting Top {config.FINAL_SELECTION_COUNT} Diverse Problems ---")
    
    input_path = os.path.join(config.OUTPUT_DIR, config.FILE_STEP5_MATRIX)
    if not os.path.exists(input_path):
        print(f"[Error] Matrix file not found: {input_path}. Run Step 05 first.")
        return

    print(f"    Loading matrices from {input_path}...")
    precomputed_data = np.load(input_path)
    
    available_keys = [k for k in precomputed_data.files if k.endswith('_matrix')]
    print(f"    Found metrics for {len(available_keys)} problems.")

    question_scores = []
    for key in tqdm(available_keys, desc="Scoring Problems"):
        q_id = int(key.split('_')[1]) 
        matrix = precomputed_data[key]
        
        score = calculate_diversity_metric(matrix)
        question_scores.append({
            "q_id": q_id,
            "score": score
        })

    df_scores = pd.DataFrame(question_scores)
    df_ranked = df_scores.sort_values(by="score", ascending=False)
    
    top_n_df = df_ranked.head(config.FINAL_SELECTION_COUNT)
    selected_ids = sorted(top_n_df['q_id'].tolist())
    
    print(f"\n    Top {len(selected_ids)} problems selected.")
    print(f"    Score Range: {top_n_df['score'].max():.4f} (Max) -> {top_n_df['score'].min():.4f} (Min)")

    output_path = os.path.join(config.OUTPUT_DIR, config.FILE_STEP6_SELECTED_Q)
    output_data = {
        "ranking_metric": "max_mean_dist",
        "selection_count": len(selected_ids),
        "selected_ids": selected_ids
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"    Saved selected IDs to: {output_path}")
    print("--- [Step 6 Complete] ---")

if __name__ == "__main__":
    select_top_problems()