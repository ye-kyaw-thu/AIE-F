#!/bin/bash
# Split shuffled_cleaned_data_4.csv into train (2/3) and test (1/3)

INPUT="${1:-data/cleaned_burmese_emotion_data.csv}"
TRAIN_OUT="${2:-data/train_data.csv}"
TEST_OUT="${3:-data/test_data.csv}"

# Count total lines
TOTAL=$(wc -l < "$INPUT")
DATA_ROWS=$((TOTAL - 1))  # Exclude header

# Calculate split
TRAIN_ROWS=$((DATA_ROWS * 5 / 6))
TEST_ROWS=$((DATA_ROWS - TRAIN_ROWS))

echo "[*] Total rows: $DATA_ROWS"
echo "[*] Train rows: $TRAIN_ROWS (2/3)"
echo "[*] Test rows:  $TEST_ROWS (1/3)"

# First file: header + 2/3 of data
head -n $((TRAIN_ROWS + 1)) "$INPUT" > "$TRAIN_OUT"

# Second file: header + remaining 1/3 of data
(head -n 1 "$INPUT"; tail -n "$TEST_ROWS" "$INPUT") > "$TEST_OUT"

echo "[*] Created: $TRAIN_OUT"
echo "[*] Created: $TEST_OUT"
