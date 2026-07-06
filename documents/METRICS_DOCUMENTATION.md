# Evaluation Metrics Documentation

## Overview
Your cardiac risk prediction model now includes **11 comprehensive evaluation metrics** for thorough performance assessment. These metrics are calculated in both `train_model.py` and `compare_models.py`.

---

## Metrics Included

### 1. **Accuracy**
- **Definition**: Percentage of correct predictions out of total predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Use**: Overall correctness, but can be misleading with imbalanced data

### 2. **Precision**
- **Definition**: Of all positive predictions, what percentage were correct?
- **Formula**: TP / (TP + FP)
- **Use**: Important when false positives are costly
- **Medical context**: Minimizes unnecessary treatments

### 3. **Recall (Sensitivity)**
- **Definition**: Of all actual positives, what percentage did we catch?
- **Formula**: TP / (TP + FN)
- **Use**: Critical for medical diagnosis - missing a patient at risk is dangerous
- **Medical context**: Don't miss cardiac risk cases

### 4. **Specificity**
- **Definition**: Of all actual negatives, what percentage did we correctly identify?
- **Formula**: TN / (TN + FP)
- **Use**: Complement to recall; shows false alarm rate
- **Medical context**: Confidence in "low-risk" classification

### 5. **F1-Score**
- **Definition**: Harmonic mean of Precision and Recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use**: Balanced metric when you care about both false positives and false negatives
- **Range**: 0 to 1 (higher is better)

### 6. **Balanced Accuracy**
- **Definition**: Average of recall and specificity
- **Formula**: (Sensitivity + Specificity) / 2
- **Use**: Better than accuracy for imbalanced datasets
- **Advantage**: Not affected by class imbalance

### 7. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
- **Definition**: Probability that model ranks a random positive case higher than a random negative case
- **Range**: 0 to 1 (0.5 = random, 1.0 = perfect)
- **Use**: Threshold-independent performance metric
- **Medical standard**: Industry standard for binary classification

### 8. **PR-AUC (Precision-Recall Area Under Curve)**
- **Definition**: Area under the precision-recall curve
- **Range**: 0 to 1 (higher is better)
- **Use**: Better than ROC-AUC for imbalanced datasets
- **Medical context**: Focuses on positive class performance

### 9. **Matthews Correlation Coefficient (MCC)**
- **Definition**: Correlation between predicted and actual classifications
- **Range**: -1 to +1 (-1 = opposite, 0 = random, +1 = perfect)
- **Use**: Single score accounting for all four confusion matrix elements
- **Advantage**: Better than accuracy for imbalanced data

### 10. **Cohen's Kappa**
- **Definition**: Measures agreement between two raters (accounting for chance)
- **Range**: -1 to +1 (higher is better)
- **Interpretation**:
  - 0.00–0.20: Slight agreement
  - 0.21–0.40: Fair agreement
  - 0.41–0.60: Moderate agreement
  - 0.61–0.80: Substantial agreement
  - 0.81–1.00: Almost perfect agreement

### 11. **Confusion Matrix**
- **Components**: 
  - **TP (True Positives)**: Correctly identified cardiac risk cases
  - **TN (True Negatives)**: Correctly identified low-risk cases
  - **FP (False Positives)**: Incorrectly flagged as high-risk
  - **FN (False Negatives)**: Missed cardiac risk cases (MOST DANGEROUS)
- **Use**: Understand the type of errors the model makes

---

## Output Files

### train_model.py generates:
1. **outputs/model_evaluation_metrics.csv** - All metrics in tabular format
2. **outputs/training_evaluation.png** - 4-panel visualization:
   - Training accuracy progression
   - Loss reduction curve
   - Confusion matrix heatmap
   - ROC-AUC curve

### compare_models.py generates:
1. **outputs/model_comparison.csv** - Side-by-side model comparison
2. **outputs/comprehensive_model_comparison.png** - 4-panel visualization:
   - Key metrics comparison (bar chart)
   - Advanced metrics comparison
   - ROC-AUC curves for all models
   - Overall quality score

---

## Interpretation Guide for Cardiac Risk Prediction

| Metric | Target Value | Why It Matters |
|--------|--------------|----------------|
| **Recall** | > 0.95 | Missing cardiac risk is dangerous |
| **Specificity** | > 0.85 | Too many false alarms waste resources |
| **ROC-AUC** | > 0.90 | Medical standard benchmark |
| **Precision** | > 0.80 | Avoid unnecessary treatments |
| **Balanced Accuracy** | > 0.85 | Overall fairness for both classes |

---

## Medical Context

For a **cardiac risk prediction model**:
- **High Recall is CRITICAL**: Missing a person with heart disease (False Negative) is dangerous
- **High Specificity is Important**: Too many false alarms reduce trust in the system
- **F1-Score Balance**: Balances the trade-off between these two

**Priority Order for Medical AI**:
1. Recall (Sensitivity) - Catch all at-risk patients
2. ROC-AUC - Overall discrimination ability
3. Specificity - Minimize false alarms
4. Precision - Avoid unnecessary treatments
5. F1-Score - Overall balance

---

## Usage Examples

### Run complete model training with all metrics:
```bash
python src/train_model.py
```
Output: CSV file + 4-panel evaluation chart

### Compare all models:
```bash
python src/compare_models.py
```
Output: Comparison CSV + comprehensive 4-panel comparison chart

---

## Glossary

- **TP/TN/FP/FN**: Components of confusion matrix
- **ROC**: Receiver Operating Characteristic
- **AUC**: Area Under Curve
- **PR-AUC**: Precision-Recall AUC
- **MCC**: Matthews Correlation Coefficient
- **Threshold**: Decision boundary (default 0.5 for probabilities)
