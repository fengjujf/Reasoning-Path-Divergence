#!/bin/bash

set -e

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

START_TIME=$(date +%s)

echo "========================================================"
echo "   RPD Data Curation Pipeline"
echo "   Start Time: $(date)"
echo "   Logs will be saved to: $LOG_DIR"
echo "========================================================"

# --- Step 1: Data Preparation ---
echo ""
echo "[Step 1/7] Downloading and Reorganizing Dataset..."
echo "---------------------------------------------------"

python 01_prepare_dataset.py 2>&1 | tee "$LOG_DIR/step01.log"

# --- Step 2: Length Filtering ---
echo ""
echo "[Step 2/7] Filtering by Length..."
echo "---------------------------------------------------"
python 02_filter_length.py 2>&1 | tee "$LOG_DIR/step02.log"

# --- Step 3: Quality Filtering (GPU) ---
echo ""
echo "[Step 3/7] Filtering by Quality (Requires GPU)..."
echo "---------------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Starting vLLM inference..."
else
    echo "WARNING: No GPU detected via nvidia-smi. This step might be slow or fail if vLLM requires CUDA."
fi
python 03_filter_quality.py 2>&1 | tee "$LOG_DIR/step03.log"

# --- Step 4: CoT Summarization (GPU) ---
echo ""
echo "[Step 4/7] Generating CoT Summaries (Requires GPU)..."
echo "---------------------------------------------------"
python 04_generate_summary.py 2>&1 | tee "$LOG_DIR/step04.log"

# --- Step 5: Distance Matrix Computation (GPU/CPU) ---
echo ""
echo "[Step 5/7] Computing RPD Distance Matrices..."
echo "---------------------------------------------------"
python 05_compute_matrix.py 2>&1 | tee "$LOG_DIR/step05.log"

# --- Step 6: Problem Selection ---
echo ""
echo "[Step 6/7] Selecting Top Diverse Problems..."
echo "---------------------------------------------------"
python 06_select_problems.py 2>&1 | tee "$LOG_DIR/step06.log"

# --- Step 7: Answer Selection & Final Assembly ---
echo ""
echo "[Step 7/7] Selecting Answers & Building Final Dataset..."
echo "---------------------------------------------------"
python 07_select_answers.py 2>&1 | tee "$LOG_DIR/step07.log"

# --- Summary ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

H=$((DURATION / 3600))
M=$(( (DURATION % 3600) / 60 ))
S=$((DURATION % 60))

echo ""
echo "========================================================"
echo "   Pipeline Completed Successfully!"
echo "   Total Duration: ${H}h ${M}m ${S}s"
echo "   Final Output: Check config.OUTPUT_DIR for results."
echo "========================================================"