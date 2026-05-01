#!/bin/bash

INPUT="sentiment_data.csv"
TRAIN="train_data.csv"
TEST="test_data.csv"
PERCENT=80

HEADER=$(head -n 1 "$INPUT")

tail -n +2 "$INPUT" | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>)' > body_shuffled.tmp

TOTAL_LINES=$(wc -l < body_shuffled.tmp)
TRAIN_LINES=$(( TOTAL_LINES * PERCENT / 100 ))

head -n "$TRAIN_LINES" body_shuffled.tmp >> "$TRAIN"

tail -n +$(( TRAIN_LINES + 1 )) body_shuffled.tmp >> "$TEST"

rm body_shuffled.tmp