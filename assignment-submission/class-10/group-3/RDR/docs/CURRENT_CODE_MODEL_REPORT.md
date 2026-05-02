# Dataset and Model Report

Generated on: 2026-04-30

## 1. Overview

This report summarizes the current dataset and the performance of the SCRDR-based snake classification model used in this project.

## 2. Dataset Summary

The dataset contains `18` records and `6` columns.

Columns:

- `species_name`
- `is_venomous`
- `head_shape`
- `eye_pupil`
- `body_pattern`
- `habitat`

Data types:

- `species_name`: categorical/text
- `is_venomous`: boolean
- `head_shape`: categorical/text
- `eye_pupil`: categorical/text
- `body_pattern`: categorical/text
- `habitat`: categorical/text

Target variable:

- `species_name`

Number of unique snake species:

- `18`

This means each row represents one species example in the current dataset.

## 3. Feature Distribution

### 3.1 Venom Status

- Venomous: `6`
- Non-venomous: `12`

### 3.2 Head Shape

- Oval: `13`
- Triangular: `5`

### 3.3 Eye Pupil

- Round: `11`
- Vertical: `7`

### 3.4 Body Pattern

- Solid: `5`
- Banded: `5`
- Blotched: `5`
- Striped: `2`
- Reticulated: `1`

### 3.5 Habitat

- Ground: `11`
- Trees: `4`
- Water: `2`
- Urban: `1`

## 4. Class Distribution

The dataset currently has one instance for each species, so the class distribution is perfectly even in count, but very small in size.

Species list:

- Monocled Cobra
- King Cobra
- Malayan Pit Viper
- White-lipped Pit Viper
- Oriental Rat Snake
- Reticulated Python
- Burmese Python
- Golden Tree Snake
- Red-necked Keelback
- Checkered Keelback
- Small-spotted Coral Snake
- Indo-Chinese Sand Snake
- Mock Viper
- Many-spotted Cat Snake
- Green Cat Snake
- Common Wolf Snake
- Laotian Wolf Snake
- Sunbeam Snake

## 5. Current Model Summary

The current model is a rule-based SCRDR classifier.

Model snapshot from the current rule base:

- Total nodes in the rule tree: `20`
- Learned rules excluding the default root: `19`
- Leaf nodes: `8`
- Maximum rule depth: `4`

Feature usage in rule conditions:

- `body_pattern`: `9`
- `habitat`: `4`
- `is_venomous`: `3`
- `head_shape`: `2`
- `eye_pupil`: `1`

This suggests that `body_pattern` is the most frequently used feature in the current model.

## 6. Current Evaluation Result

Evaluation was run on the current dataset using the saved model.

Result summary:

- Total samples tested: `18`
- Correct predictions: `17`
- Incorrect predictions: `1`
- Accuracy: `94.4%`

Overall classification metrics:

- Macro precision: `0.92`
- Macro recall: `0.94`
- Macro F1-score: `0.93`
- Weighted precision: `0.92`
- Weighted recall: `0.94`
- Weighted F1-score: `0.93`

Most species achieved perfect classification on the current dataset.

## 7. Observed Error

The model made one classification error:

- Actual class: `Sunbeam Snake`
- Predicted class: `Oriental Rat Snake`

Feature values for the misclassified sample:

```text
is_venomous = False
head_shape = Oval
eye_pupil = Round
body_pattern = Solid
habitat = Ground
```

This shows that the current rules do not yet clearly separate `Sunbeam Snake` from another non-venomous snake with similar feature values.

## 8. Interpretation

The current model performs well on the available dataset, reaching `94.4%` accuracy. However, the dataset is very small, with only `18` examples and only one example per species. Because of this, the result is useful as a classroom demonstration, but it is not strong evidence of real-world generalization.

The model appears to rely most heavily on visible categorical traits such as `body_pattern`, `habitat`, and `is_venomous`. This is appropriate for an interpretable rule-based system, especially in an educational setting.

## 9. Limitations

1. The dataset is very small.
2. Each species has only one sample.
3. Evaluation was performed on the same dataset used for rule construction and testing.
4. The current report reflects in-project performance, not unseen external data performance.

## 10. Suggested Improvements

1. Add more samples for each snake species.
2. Add new rules to separate `Sunbeam Snake` from similar non-venomous classes.
3. Prepare a separate test set for more reliable evaluation.
4. Expand the feature set if finer-grained species distinctions are needed.
