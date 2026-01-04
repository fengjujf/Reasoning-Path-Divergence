import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
import config  

def download_and_process_dataset():
    print(f"--- [Step 1] Processing files from Index {config.START_FILE_INDEX} to {config.END_FILE_INDEX} ---")
    
    question_data = {}
    processing_range = range(config.START_FILE_INDEX, config.END_FILE_INDEX + 1)
    
    for i in tqdm(processing_range, desc="Downloading & Reading files"):
   
        repo_filename = config.SOURCE_FILENAME_FMT.format(index=i)
        
        try:
           
            local_file_path = hf_hub_download(
                repo_id=config.SOURCE_DATASET_REPO,
                filename=repo_filename,
                repo_type="dataset",
                local_dir=config.RAW_DATA_DIR,
                local_dir_use_symlinks=False 
            )
            
    
            df = pd.read_parquet(local_file_path, columns=['conversations', 'difficulty'])
            
            for _, row in df.iterrows():
                conversations = row['conversations']
                difficulty = row['difficulty']
                
                human_value = next((msg['value'] for msg in conversations if msg['from'] == 'human'), None)
                gpt_value = next((msg['value'] for msg in conversations if msg['from'] == 'gpt'), None)
                
                if human_value and gpt_value:
                    if human_value not in question_data:
                        question_data[human_value] = {'answers': [], 'difficulty': difficulty}
                    question_data[human_value]['answers'].append(gpt_value)
                    
        except Exception as e:
          
            print(f"\n[Warning] Failed to process index {i} ({repo_filename}): {e}")
            continue

    print(f"\n--- [Step 1 Complete] Collected {len(question_data)} unique questions ---")

  
    process_and_save_reorganized(question_data)

def process_and_save_reorganized(question_data):
    print("--- [Step 2] Filtering (must have 16 answers) and Saving ---")
    
   
    filtered_questions = {k: v for k, v in question_data.items() if len(v['answers']) == 16}
    print(f"    Qualified questions found: {len(filtered_questions)}")

    if not filtered_questions:
        print("    No data met the criteria. Exiting.")
        return

  
    schema_fields = [pa.field('question', pa.string())]
    for k in range(16):
        schema_fields.append(pa.field(f'answer_{k}', pa.string()))
    schema_fields.append(pa.field('difficulty', pa.int64()))
    output_schema = pa.schema(schema_fields)

    new_records = []
    file_counter = 0

    for question, data in tqdm(filtered_questions.items(), desc="Writing reorganized files"):
        record = {
            'question': question,
            'difficulty': data['difficulty']
        }
        for k, ans in enumerate(data['answers']):
            record[f'answer_{k}'] = ans
            
        new_records.append(record)

        if len(new_records) >= config.TARGET_ROWS_PER_FILE:
            save_chunk(new_records, file_counter, output_schema)
            new_records = []
            file_counter += 1

    if new_records:
        save_chunk(new_records, file_counter, output_schema)

def save_chunk(records, counter, schema):
    filename = f"reorganized_data_{counter:05d}.parquet"
    path = os.path.join(config.PROCESSED_DATA_DIR, filename)
    
    df_output = pd.DataFrame(records)
    table = pa.Table.from_pandas(df_output, schema=schema, preserve_index=False)
    pq.write_table(table, path)
    print(f"    Saved file: {path} ({len(records)} records)")

if __name__ == "__main__":
    download_and_process_dataset()