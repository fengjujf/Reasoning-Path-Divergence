import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import config  


try:
    import tiktoken
    TIKTOKEN_ENABLED = True
except ImportError:
    print("[Fatal Error] 'tiktoken' library not found. Please run 'pip install tiktoken'.")
    exit(1)

def count_tokens(text, tokenizer):
    """使用 tiktoken 计算文本 Token 数"""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return 0

    return len(tokenizer.encode(text, allowed_special="all"))

def filter_dataset_by_length():
    print(f"--- [Step 2] Filtering by Average Token Length < {config.MAX_AVG_TOKENS} ---")

    input_files = sorted(glob.glob(os.path.join(config.PROCESSED_DATA_DIR, "reorganized_data_*.parquet")))
    if not input_files:
        print(f"[Error] No input files found in {config.PROCESSED_DATA_DIR}. Did you run Step 01?")
        return


    print("    Loading tokenizer (cl100k_base)...")
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"    [Error] Failed to load tokenizer: {e}")
        return

    filtered_buffer = []
    file_counter = 0
    total_processed = 0
    total_kept = 0
    
    sample_df = pd.read_parquet(input_files[0])
    output_schema = pa.Schema.from_pandas(sample_df, preserve_index=False)


    for file_path in tqdm(input_files, desc="Filtering files"):
        try:
            df = pd.read_parquet(file_path)
            current_rows = len(df)
            total_processed += current_rows
            
      
            ans_cols = [c for c in df.columns if c.startswith('answer_')]
            
 
            token_counts = pd.DataFrame(index=df.index)
            
            for col in ans_cols:
      
                token_counts[col] = df[col].apply(lambda x: count_tokens(x, tokenizer))
            
         
            total_tokens = token_counts.sum(axis=1)
            valid_answers = df[ans_cols].notna().sum(axis=1)
            
            avg_tokens = total_tokens / valid_answers.replace(0, 1)
            
            mask = avg_tokens < config.MAX_AVG_TOKENS
            df_filtered = df[mask].copy()
            
            if not df_filtered.empty:
                filtered_buffer.extend(df_filtered.to_dict('records'))
                total_kept += len(df_filtered)
            
          
            while len(filtered_buffer) >= config.TARGET_ROWS_PER_FILE:
                
                chunk_to_save = filtered_buffer[:config.TARGET_ROWS_PER_FILE]
                filtered_buffer = filtered_buffer[config.TARGET_ROWS_PER_FILE:]
                
                save_chunk(chunk_to_save, file_counter, output_schema)
                file_counter += 1

        except Exception as e:
            print(f"    [Warning] Error processing file {file_path}: {e}")
            continue

    
    if filtered_buffer:
        save_chunk(filtered_buffer, file_counter, output_schema)
        file_counter += 1

    print(f"\n--- [Step 2 Complete] ---")
    print(f"    Processed: {total_processed} rows")
    print(f"    Kept:      {total_kept} rows ({(total_kept/total_processed if total_processed else 0):.2%})")
    print(f"    Removed:   {total_processed - total_kept} rows")

def save_chunk(records, counter, schema):
    filename = f"length_filtered_data_{counter:05d}.parquet"
    path = os.path.join(config.PROCESSED_DATA_DIR, filename)
    
    df = pd.DataFrame(records)
   
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)
  
if __name__ == "__main__":
    filter_dataset_by_length()