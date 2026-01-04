import os


SOURCE_DATASET_REPO = "open-thoughts/OpenThoughts3-1.2M"
MODEL_PATH_INSTRUCT="Qwen/Qwen3-14B"
MODEL_PATH_EMBEDDING="Qwen/Qwen3-Embedding-8B"
SOURCE_FILENAME_FMT = "data/train-{index:05d}-of-00120.parquet"

START_FILE_INDEX = 25
END_FILE_INDEX = 109

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw_source")
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, "processed")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

TARGET_ROWS_PER_FILE = 10000 
MAX_AVG_TOKENS=14000
MIN_HIGH_QUALITY_ANSWERS=10



# [Step 4] 
DIR_STEP4_SUMMARY = os.path.join(PROCESSED_DATA_DIR, "summaries")
os.makedirs(DIR_STEP4_SUMMARY, exist_ok=True)

# [Step 5, 6, 7] 
OUTPUT_DIR = PROCESSED_DATA_DIR
FILE_STEP5_MATRIX = "05_distance_matrix.npz"
FILE_STEP6_SELECTED_Q = "06_selected_questions.json"
FILE_STEP7_FINAL = "07_final_dataset.json"

# [Step 6 & 7] 
FINAL_SELECTION_COUNT = 100
ANSWERS_PER_QUESTION = 3



# ================= TEST =================
TEST_LIMIT = None