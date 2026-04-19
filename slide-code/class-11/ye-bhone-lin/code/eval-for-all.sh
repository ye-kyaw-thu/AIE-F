#!/bin/bash

GT="/Users/yebhonelin/Documents/LU_Lab/myAEC/code-and-config/notebook-examples/evaluation_code/gt.txt"
ASR="/Users/yebhonelin/Documents/LU_Lab/myAEC/code-and-config/notebook-examples/evaluation_code/err.txt"
LOG="/Users/yebhonelin/Documents/LU_Lab/myAEC/code-and-config/notebook-examples/evaluation_code/metrics.log"

# Check if required files exist
if [ ! -f "$GT" ]; then
    echo "GT file not found: $GT"
    exit 1
fi

if [ ! -f "$ASR" ]; then
    echo "ASR file not found: $ASR"
    exit 1
fi

# Clear or create the log file
> "$LOG"

# Count lines in ground truth file
NUM_LINES=$(wc -l < "$GT")

for ((i=1; i<=NUM_LINES; i++)); do
    GT_LINE=$(sed -n "${i}p" "$GT")
    ASR_LINE=$(sed -n "${i}p" "$ASR")

    echo "Evaluating Line $i" >> "$LOG"
    echo "GT : $GT_LINE" >> "$LOG"
    echo "ASR: $ASR_LINE" >> "$LOG"

    echo "$GT_LINE" > __gt_line.txt
    echo "$ASR_LINE" > __hyp_line.txt

    python /Users/yebhonelin/Documents/LU_Lab/myAEC/code-and-config/notebook-examples/evaluation_code/cal_err_scores_2.py __gt_line.txt __hyp_line.txt >> "$LOG" 2>&1
    python /Users/yebhonelin/Documents/LU_Lab/myAEC/code-and-config/notebook-examples/evaluation_code/chrF++.py -R __gt_line.txt -H __hyp_line.txt >> "$LOG" 2>&1
    python /Users/yebhonelin/Documents/LU_Lab/myAEC/code-and-config/notebook-examples/evaluation_code/8eval.py --level corpus --reference __gt_line.txt --hypothesis __hyp_line.txt >> "$LOG" 2>&1

    echo "===============<Line $i>===============" >> "$LOG"
done

# Clean up temporary files
rm -f __gt_line.txt __hyp_line.txt

echo "Finished evaluating $NUM_LINES lines"
echo "Log saved to: $LOG"
