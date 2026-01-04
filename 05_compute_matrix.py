import os
import glob
import json
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import config  
import utils   



def load_embedding_model():
    print(f"--- Loading Embedding Model: {config.MODEL_PATH_EMBEDDING} ---")
    try:
      
        model = SentenceTransformer(config.MODEL_PATH_EMBEDDING, trust_remote_code=True)
        model.eval()
        return model
    except Exception as e:
        print(f"[Fatal Error] Failed to load embedding model: {e}")
        exit(1)

def compute_distance_matrices():
    print(f"--- [Step 5] Computing Pairwise Distance Matrices (RPD Metric) ---")
    

    summary_dir = config.DIR_STEP4_SUMMARY
    summary_files = sorted(glob.glob(os.path.join(summary_dir, "*.json")))
    
    if not summary_files:
        print(f"[Error] No summary files found in {summary_dir}. Run Step 04 first.")
        return
    if config.TEST_LIMIT:
        print(f"[TEST MODE] Limiting processing to first {config.TEST_LIMIT} files")
        summary_files = summary_files[:config.TEST_LIMIT]
    
    model = load_embedding_model()
    
    
    data_to_save = {}
    valid_problems_count = 0
    
    
    for file_path in tqdm(summary_files, desc="Processing Problems"):
        try:

            file_name = os.path.basename(file_path)
            q_id_str = os.path.splitext(file_name)[0]

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summaries_list = data.get("summaries", [])
            summaries_dict = {s['answer_id']: s for s in summaries_list}
            
            valid_summaries = {}
            for ans_id, summary in summaries_dict.items():
                steps = summary.get('logical_steps', [])
                if isinstance(steps, list) and steps and any(s.get('step_description') for s in steps):
                    valid_summaries[ans_id] = summary
            
            if len(valid_summaries) < 2:
                continue

            texts_to_embed = set()
            for summary in valid_summaries.values():
                for step in summary['logical_steps']:
                    desc = step.get('step_description')
                    if desc and isinstance(desc, str) and desc.strip():
                        texts_to_embed.add(desc)
            
            if not texts_to_embed:
                continue
                
            unique_texts = list(texts_to_embed)
            embeddings = model.encode(unique_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
           
            embeddings_dict = {text: emb for text, emb in zip(unique_texts, embeddings)}
           
            valid_ids = sorted(list(valid_summaries.keys()))
            n_samples = len(valid_ids)
            dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples): 
                    id_a = valid_ids[i]
                    id_b = valid_ids[j]
                    
                    summary_a = valid_summaries[id_a]
                    summary_b = valid_summaries[id_b]
                    
                    dist = utils.calculate_rpd_distance(summary_a, summary_b, embeddings_dict)
                    
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
            
            data_to_save[f"q_{q_id_str}_matrix"] = dist_matrix
            data_to_save[f"q_{q_id_str}_ids"] = np.array(valid_ids)
            
            valid_problems_count += 1
            
        except Exception as e:
            print(f"    [Warning] Error processing file {file_path}: {e}")
            continue

    output_path = os.path.join(config.OUTPUT_DIR, config.FILE_STEP5_MATRIX)
    print(f"\nSaving data for {valid_problems_count} problems to {output_path}...")
    np.savez_compressed(output_path, **data_to_save)
    print("--- [Step 5 Complete] ---")

if __name__ == "__main__":
    compute_distance_matrices()