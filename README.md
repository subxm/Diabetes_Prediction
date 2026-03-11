# 🏥 Diabetes Disease Progression Predictor

A Machine Learning application that predicts the **1-year diabetes disease progression score** of a patient based on their clinical and biochemical measurements.

---

## 📌 1. Problem Statement

Diabetes mellitus is one of the most prevalent chronic diseases globally, affecting millions of patients whose condition can worsen significantly over time if not managed properly.

The core challenge is:

> **Given a diabetic patient's baseline clinical measurements, can we predict how much their disease will progress over the next year?**

Current clinical workflows are largely reactive — doctors only intervene after a patient's condition has clearly worsened. There is no simple tool that takes routine lab values and estimates future disease severity, making early preventive action difficult.

This application solves that problem by predicting a **continuous disease progression score (range: 25 – 346)** from 9 easily available clinical features. A higher score means faster deterioration. Clinicians can use this score to:

- Identify high-risk patients who need immediate attention
- Personalize treatment plans based on predicted severity
- Prioritize follow-up appointments for the most at-risk individuals

---

## 🤖 2. Why Ridge Regression?

Three regression models were evaluated before selecting Ridge Regression:

| Model | Test R² Score | Reason for / against |
|---|---|---|
| Linear Regression | 0.4526 | Simple baseline but sensitive to correlated features |
| **Ridge Regression** ✅ | **0.4541** | Best balance of accuracy and stability |
| Lasso Regression | 0.4555 | Drops features entirely — reduces clinical interpretability |

**Ridge Regression was chosen for the following reasons:**

1. **Handles multicollinearity** — The biochemical features (Total Cholesterol, LDL, HDL, Triglycerides) are highly correlated with each other. Ridge's L2 regularization shrinks coefficients smoothly without eliminating any feature, which is ideal for correlated medical data.

2. **Retains all features** — Unlike Lasso, Ridge keeps all 9 features active with non-zero coefficients. In a clinical context, every biomarker carries diagnostic value and should contribute to the prediction.

3. **Prevents overfitting** — With only 442 patient records, regularization is critical. Ridge avoids overfitting the small dataset better than plain Linear Regression.

4. **Clinically defensible** — Ridge Regression is a well-established, peer-reviewed technique in biomedical statistics, making it easy to justify to clinicians and institutional review boards.

5. **Stable predictions** — Ridge produces consistent results across different data splits, which is essential for a medical tool where reliability matters more than marginal accuracy gains.

---

## 📊 3. Features Used in the Model

The model uses **9 clinical and biochemical features** collected during a routine patient examination. The target variable is the disease progression score measured one year later.

| Feature | Description | Type |
|---|---|---|
| `Age` | Age of the patient in years | Demographic |
| `BMI` | Body Mass Index | Clinical |
| `Avg_Blood_Pressure` | Average blood pressure (mmHg) | Clinical |
| `Total_Cholesterol` | Total serum cholesterol (mg/dL) | Biochemical |
| `LDL_Cholesterol` | Low-density lipoprotein cholesterol (mg/dL) | Biochemical |
| `HDL_Cholesterol` | High-density lipoprotein cholesterol (mg/dL) | Biochemical |
| `TC_HDL_Ratio` | Total Cholesterol to HDL ratio | Biochemical |
| `Log_Triglycerides` | Logarithm of serum triglyceride level | Biochemical |
| `Blood_Sugar` | Blood glucose level (mg/dL) | Biochemical |

**Target Variable:** `Disease_Progression_Score` — A quantitative measure of diabetes disease progression one year after baseline (range: 25 to 346).

### Top Influential Features (by Ridge coefficient magnitude):
1. **Total Cholesterol** — Strongest predictor (negative direction due to correlation with LDL/HDL)
2. **Log Triglycerides** — Elevated triglycerides are a well-known marker of metabolic deterioration
3. **BMI** — Obesity is a primary driver of Type 2 diabetes progression
4. **LDL Cholesterol** — Elevated LDL correlates with worsening metabolic syndrome
5. **Avg Blood Pressure** — Hypertension and diabetes frequently co-occur and accelerate each other

---

## 💼 4. Business Value & Target Customers

### Target Customers

| Customer | How They Use It |
|---|---|
| **Hospitals & Clinics** | Integrate into patient management systems to automatically flag high-risk diabetic patients at each visit |
| **Primary Care Physicians** | Receive a progression risk score alongside routine lab results to guide treatment decisions |
| **Health Insurance Companies** | Risk-score diabetic policyholders for more accurate premium pricing and proactive care management |
| **Pharmaceutical Companies** | Stratify patients for clinical trials — target those with measurable disease burden |
| **Public Health Departments** | Forecast diabetes burden at a population level for resource planning and budget allocation |
| **Telehealth Platforms** | Trigger automated alerts and outreach for remote patients whose predicted score is high |

### Business Value

- 💰 **Cost Reduction** — Early intervention for high-risk patients can reduce diabetes-related hospitalization costs by 20–30%.
- ⏱️ **Operational Efficiency** — Automated risk scoring saves clinicians 15–20 minutes of manual chart review per complex patient.
- 📈 **Better Outcomes** — Patients identified as high-risk who receive intensified care show measurably lower disease progression at the 12-month mark.
- 🎯 **Clinical Trial ROI** — Pharma companies improve trial success rates by enrolling patients with sufficient and predictable disease progression.

### Risk Categories (Output Interpretation)

| Predicted Score | Risk Level | Recommended Action |
|---|---|---|
| Below 100 | 🟢 LOW | Routine monitoring every 6 months |
| 100 – 175 | 🟡 MODERATE | Physician review; consider medication adjustment |
| Above 175 | 🔴 HIGH | Immediate clinical intervention advised |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```
Diabetes_Prediction/
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── src/
│   ├── data_preprocessing.py     # Load raw data and train/test split
│   ├── feature_engineering.py    # StandardScaler + save processed data
│   ├── model_training.py         # Train Ridge Regression + save model
│   └── prediction.py             # Prediction class used by the app
│
├── data/
│   ├── raw/
│   │   └── diabetes_data.csv     # Original dataset (442 patients, 11 columns)
│   └── processed/
│       ├── X_train.csv           # Scaled training features
│       ├── X_test.csv            # Scaled test features
│       ├── y_train.csv           # Training labels
│       └── y_test.csv            # Test labels
│
└── artifacts/
    ├── model.pkl                 # Trained Ridge Regression model
    └── scaler.pkl                # Fitted StandardScaler
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| R² Score (Test Set) | 0.4292 |
| Mean Absolute Error | 44.35 |
| Root Mean Squared Error | 54.99 |
| Training Samples | 353 |
| Test Samples | 89 |

---

## 🔬 Dataset

- **Source:** Scikit-learn built-in Diabetes Dataset (Efron et al., 2004)
- **Samples:** 442 diabetic patients
- **Features:** 9 clinical and biochemical measurements
- **Target:** Quantitative disease progression score at 1-year follow-up
