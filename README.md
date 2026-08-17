# creditwise-loan-approval
ML pipeline predicting loan approval from applicant data — EDA, feature engineering, and comparison of Logistic Regression, KNN &amp; Naive Bayes (up to 88% accuracy)
# CreditWise — Loan Approval Prediction

A supervised machine learning project that predicts whether a loan application will be approved, based on applicant financial and demographic data.

## Problem

Lenders need a fast, consistent way to flag which loan applications are likely to be approved, using historical applicant data (income, credit score, debt-to-income ratio, employment, education, etc.).

## Dataset

`loan_approval_data.csv` — applicant and co-applicant income, savings, credit score, debt-to-income ratio, employment status, marital status, education level, loan purpose, property area, and a binary `Loan_Approved` target.

> Add your dataset source/link here (e.g. Kaggle) if you can share it, or a short note on how to obtain it.

## Approach

1. **Data cleaning** — imputed missing numerical values (mean) and categorical values (mode).
2. **EDA** — class balance, distributions of income/credit score, boxplots to check for outliers, and how each feature relates to loan approval.
3. **Feature encoding** — label encoding for education/target, one-hot encoding for categorical fields.
4. **Modelling** — trained and compared Logistic Regression, KNN, and Naive Bayes on a held-out 20% test set.
5. **Feature engineering** — added squared terms for credit score and debt-to-income ratio to capture non-linear effects, then re-trained all three models.

## Results

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.785 | 0.836 | 0.810 | 0.880 |
| KNN | 0.673 | 0.574 | 0.619 | 0.785 |
| Naive Bayes | 0.811 | 0.705 | 0.754 | 0.860 |

**Best model:** Naive Bayes on precision (fewest false approvals), Logistic Regression on overall accuracy and recall. Feature engineering (squared credit score / DTI ratio) improved every model over the baseline.

## Tech stack

Python, pandas, numpy, seaborn, matplotlib, scikit-learn

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Credit_wise.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`) to reproduce the results above.
