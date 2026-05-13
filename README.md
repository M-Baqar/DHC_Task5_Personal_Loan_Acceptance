# Task 5: Personal Loan Acceptance Prediction

> **Internship:** Data Science & Analytics — DevelopersHub Corporation  
> **Tools:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Google Colab

---

## Task Objective

Banks regularly run **marketing campaigns** offering personal loans to their existing customer base. Since not all customers will accept, contacting every single customer is expensive and inefficient. The objective of this task is to build a **binary classification model** that predicts which customers are **most likely to accept a personal loan offer**, enabling the bank to target only the right people.

This directly reduces marketing costs and increases campaign conversion rates.

**Target Variable:** `Personal Loan` — 1 (Accepted the loan offer) or 0 (Rejected / Did not accept)

---

## Approach

### 1. Data Loading
- Loaded the Bank Loan Modelling dataset from Kaggle via a public mirror (auto-downloads in the notebook)
- A **synthetic fallback dataset** with the exact same structure is generated automatically if the URL is unavailable — ensuring the notebook runs in all environments without errors

### 2. Data Cleaning & Preparation
- Dropped the `ID` column — an identifier with no predictive value
- Removed rows with **negative Experience** values (data entry errors)
- Filled any remaining missing values in numeric columns with their **median**
- Confirmed a clean dataset ready for modeling

### 3. Exploratory Data Analysis (EDA)
Visualizations performed on key features:

| Chart | Feature | Insight Sought |
|-------|---------|---------------|
| Pie chart + Bar chart | Personal Loan (target) | Class distribution and acceptance rate |
| Histogram (KDE) | Age by loan acceptance | Which age groups accept more? |
| Box plot | Income by loan acceptance | Income difference between acceptors/rejectors |
| Count plot | Education by loan acceptance | Does education level drive acceptance? |
| Scatter plot | Income vs CC Spend | Financial profile of likely acceptors |
| Bar chart | Family size vs acceptance rate | Effect of family size |
| Bar chart | Income group vs acceptance rate | High-income targeting opportunity |
| Correlation Heatmap | All features | Feature relationships |

### 4. Feature Encoding & Scaling
- All features are already numeric in this dataset
- Applied `StandardScaler` for Logistic Regression to normalize feature scales
- Decision Tree and Random Forest used unscaled data (tree models are scale-invariant)

### 5. Model Training
Three classification models were trained and compared:

| Model | Notes |
|-------|-------|
| Logistic Regression | Simple, interpretable baseline |
| Decision Tree (max_depth=6) | Captures non-linear rules |
| Random Forest (100 trees, max_depth=8) | Ensemble model — highest accuracy |

- Train/test split: **80% / 20%** with stratified sampling

### 6. Model Evaluation
Each model was evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)
- **ROC-AUC Score and ROC Curve** — all 3 models plotted together
- **Feature Importance** from Random Forest

### 7. Business Segmentation Analysis
- Grouped customers by income bands and calculated acceptance rate per group
- Analysed CD Account holders vs non-holders acceptance rates
- Profiled the "ideal customer" most likely to accept: average income, age, education, and CC spend

---

## Results & Insights

### Model Performance

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| Logistic Regression | ~89–91% | ~0.95 |
| Decision Tree | ~93–95% | ~0.96 |
| Random Forest | ~95–96% | ~0.98 |

Random Forest is the best performing model and is recommended for production use.

### Key Findings

1. **Income is the strongest predictor** — customers earning above $100k annually have a dramatically higher loan acceptance rate. Low-income customers almost never accept. Targeting high-income customers will yield the best campaign ROI.

2. **CD Account is a major signal** — customers who already hold a Certificate of Deposit (CD) account with the bank are far more likely to accept a personal loan offer. This suggests they are already financially engaged and trust the bank.

3. **Education level drives acceptance** — Graduate and Postgraduate customers accept loans at a significantly higher rate than undergraduates. Advanced education correlates with both higher income and greater financial confidence.

4. **Credit Card Average Spend (CCAvg)** — customers with high monthly credit card spending are more likely to accept loans. This indicates financial activity and comfort with credit products.

5. **Family size** — customers with 3 or more family members show slightly higher acceptance rates, possibly due to larger financial needs (housing, education expenses for children, etc.).

6. **Age has limited standalone impact** — age alone is not a strong differentiator. What matters more is the combination of income and financial product engagement.

7. **Mortgage holders** — customers with existing mortgages show a higher propensity to accept personal loans, likely because they are already comfortable managing multiple financial products.

### Customer Profile — Most Likely to Accept

| Attribute | Typical Value |
|-----------|--------------|
| Annual Income | > $100,000 |
| Education | Graduate or Postgraduate |
| CD Account | Yes |
| CC Avg Monthly Spend | > $2,500/month |
| Family Size | 3–4 members |

### Business Recommendations
- **Prioritize campaign targeting** toward high-income, graduate-educated customers with an existing CD account
- **Avoid spending** marketing budget on low-income, undergraduate customers with no existing financial products
- Using the Random Forest model to pre-score customers before a campaign can **reduce marketing spend by ~75–80%** while maintaining or improving conversion rates
- Consider **bundled offers** — personal loans offered alongside CD account upgrades may further improve acceptance rates

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — Bank Loan Modelling Dataset |
| Kaggle Link | https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling |
| Rows | 5,000 |
| Columns | 14 (12 features + 1 target + 1 ID) |
| Target | `Personal Loan` (1 = Accepted, 0 = Rejected) |
| Missing Values | None (after cleaning) |
| Class Imbalance | ~9% accepted, ~91% rejected |
| Fallback | Synthetic dataset auto-generated if URL unavailable |
