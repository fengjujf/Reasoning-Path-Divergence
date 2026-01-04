import json
import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import config

def extract_boxed_json(raw_text):
    
    if not isinstance(raw_text, str):
        return None

    start_marker = '//boxed{'
    start_pos = raw_text.find(start_marker)
    
    if start_pos == -1:
        start_marker = 'boxed{'
        start_pos = raw_text.find(start_marker)
        if start_pos == -1:
            return None
    
    json_start_pos = start_pos + len(start_marker) - 1 
    
    brace_level = 0
    json_end_pos = -1
    
    for i in range(json_start_pos, len(raw_text)):
        char = raw_text[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
            
        if brace_level == 0:
            json_end_pos = i + 1
            break
            
    if json_end_pos == -1:
        return None
        
    json_str = raw_text[json_start_pos:json_end_pos]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def calculate_rpd_distance(summary_A, summary_B, embeddings_dict):
   
    steps_A = [
        s.get('step_description', '') 
        for s in summary_A.get('logical_steps', []) 
        if isinstance(s, dict) and s.get('step_description')
    ]
    steps_B = [
        s.get('step_description', '') 
        for s in summary_B.get('logical_steps', []) 
        if isinstance(s, dict) and s.get('step_description')
    ]

    if not steps_A or not steps_B:
        return 1.0 if steps_A or steps_B else 0.0

    try:
        embeds_A = np.array([embeddings_dict[s] for s in steps_A])
        embeds_B = np.array([embeddings_dict[s] for s in steps_B])
    except KeyError:
        return 1.0

    if len(embeds_A) <= len(embeds_B):
        shorter, longer = embeds_A, embeds_B
    else:
        shorter, longer = embeds_B, embeds_A

 
    cost_matrix = cdist(shorter, longer, metric='cosine')
    
    min_distances = np.min(cost_matrix, axis=1)
    
    
    return float(np.mean(min_distances))


def load_embedding_model():

    print(f"--- [Utils] Loading Embedding Model: {config.MODEL_PATH_EMBEDDING} ---")
    try:
        model = SentenceTransformer(config.MODEL_PATH_EMBEDDING, trust_remote_code=True)
        model.eval()
        
        if hasattr(model, "to"):
            import torch
            if torch.cuda.is_available():
                model.to("cuda")
                
        return model
    except Exception as e:
        print(f"[Fatal Error] Failed to load embedding model from {config.MODEL_PATH_EMBEDDING}")
        print(f"Details: {e}")
        print("Please check your 'config.py' and internet connection.")
        exit(1)