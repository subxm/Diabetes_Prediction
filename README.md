# 🏥 Diabetes Disease Progression Predictor

A Machine Learning application that predicts the **1-year diabetes disease progression score** of a patient based on their clinical and biochemical measurements using **Ridge Regression**.

🔗 **Live Demo:** [diabetes-prediction-by-subhamsingh.streamlit.app](https://diabetes-prediction-by-subhamsingh.streamlit.app/)

---

## 📌 Problem Statement

Diabetes mellitus is one of the most prevalent chronic diseases globally, affecting millions of patients whose condition can worsen significantly over time if not managed properly.

The core challenge is:

> **Given a diabetic patient's baseline clinical measurements, can we predict how much their disease will progress over the next year?**

Current clinical workflows are largely reactive — doctors only intervene after a patient's condition has clearly worsened. There is no simple tool that takes routine lab values and estimates future disease severity, making early preventive action difficult.

This application solves that problem by predicting a **continuous disease progression score (range: 25 – 346)** from 9 easily available clinical features. A higher score means faster deterioration. Clinicians can use this score to:

- Identify high-risk patients who need immediate attention
- Personalize treatment plans based on predicted severity
- Prioritize follow-up appointments for the most at-risk individuals

---

## 🤖 Why Ridge Regression?

Three regression models were evaluated before selecting Ridge Regression:

| Model | Test R² Score | Notes |
|---|---|---|
| Linear Regression | 0.4526 | Simple baseline but sensitive to correlated features |
| **Ridge Regression ✅** | **0.4541** | Best balance of accuracy and stability |
| Lasso Regression | 0.4555 | Drops features entirely — reduces clinical interpretability |

**Ridge Regression was chosen for the following reasons:**

1. **Handles multicollinearity** — Biochemical features (Total Cholesterol, LDL, HDL, Triglycerides) are highly correlated. Ridge's L2 regularization shrinks coefficients smoothly without eliminating features, which is ideal for correlated medical data.
2. **Retains all features** — Unlike Lasso, Ridge keeps all 9 features active with non-zero coefficients. In a clinical context, every biomarker carries diagnostic value and should contribute to the prediction.
3. **Prevents overfitting** — With only 442 patient records, regularization is critical. Ridge avoids overfitting better than plain Linear Regression.
4. **Clinically defensible** — Ridge Regression is a well-established, peer-reviewed technique in biomedical statistics.
5. **Stable predictions** — Produces consistent results across different data splits, essential for a medical tool where reliability matters more than marginal accuracy gains.

---

## 📊 Features Used in the Model

The model uses **9 clinical and biochemical features** collected during a routine patient examination.

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

### Top Influential Features (by Ridge coefficient magnitude)

1. **Total Cholesterol** — Strongest predictor (negative direction due to correlation with LDL/HDL)
2. **Log Triglycerides** — Elevated triglycerides are a well-known marker of metabolic deterioration
3. **BMI** — Obesity is a primary driver of Type 2 diabetes progression
4. **LDL Cholesterol** — Elevated LDL correlates with worsening metabolic syndrome
5. **Avg Blood Pressure** — Hypertension and diabetes frequently co-occur and accelerate each other

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| R² Score (Test Set) | 0.4292 |
| Mean Absolute Error | 44.35 |
| Root Mean Squared Error | 54.99 |
| Training Samples | 353 |
| Test Samples | 89 |

---

## 🟢 Risk Category Interpretation

The predicted score is automatically categorized into a risk level:

| Predicted Score | Risk Level | Recommended Action |
|---|---|---|
| Below 100 | 🟢 LOW | Routine monitoring every 6 months |
| 100 – 175 | 🟡 MODERATE | Physician review; consider medication adjustment |
| Above 175 | 🔴 HIGH | Immediate clinical intervention advised |

---

## 🔬 Dataset

- **Source:** Scikit-learn built-in Diabetes Dataset (Efron et al., 2004)
- **Total Samples:** 442 diabetic patients
- **Features:** 9 clinical and biochemical measurements
- **Target:** Quantitative disease progression score at 1-year follow-up

---

## 🏗️ Project Structure

```
Diabetes_Prediction/
│
├── app.py                        # Streamlit web application (entry point)
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

## 🔬 How It Works — Pipeline

```
Raw Clinical Input (9 features)
        ↓
Feature Scaling (StandardScaler → scaler.pkl)
        ↓
Ridge Regression Inference (model.pkl)
        ↓
Disease Progression Score (25 – 346)
        ↓
Risk Category (LOW / MODERATE / HIGH)
```

1. **Data Preprocessing** (`data_preprocessing.py`) — Loads the raw diabetes dataset and performs train/test split (80/20).
2. **Feature Engineering** (`feature_engineering.py`) — Applies `StandardScaler` to normalize features; saves the fitted scaler to `artifacts/scaler.pkl` and processed splits to `data/processed/`.
3. **Model Training** (`model_training.py`) — Trains Ridge Regression on scaled training data; saves the fitted model to `artifacts/model.pkl`.
4. **Prediction** (`prediction.py`) — `Diabetes_Prediction` class loads the saved model and scaler, scales new inputs, and returns the predicted progression score.
5. **Web App** (`app.py`) — Streamlit UI that collects 9 clinical inputs, calls the prediction class, displays the score, and shows the appropriate risk category.

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/subxm/Diabetes_Prediction.git
cd Diabetes_Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ How to Use the App

1. Open the app in your browser (locally or via the live link).
2. Enter the patient's **Age**, **BMI**, and **Average Blood Pressure**.
3. Enter biochemical values: **Total Cholesterol**, **LDL**, **HDL**, **TC/HDL Ratio**, **Log Triglycerides**, **Blood Sugar**.
4. Click **Predict**.
5. The app displays the **disease progression score** and the corresponding **risk category** (LOW / MODERATE / HIGH) with a recommended action.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Web Framework | Streamlit |
| ML Algorithm | Ridge Regression (Scikit-learn) |
| Data Handling | Pandas, NumPy |
| Model Serialization | Pickle (`.pkl`) |
| Dataset | Scikit-learn built-in Diabetes Dataset |
| Deployment | Streamlit Community Cloud |

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 💼 Use Cases & Target Audience

| User | How They Use It |
|---|---|
| **Hospitals & Clinics** | Integrate into patient management systems to automatically flag high-risk diabetic patients at each visit |
| **Primary Care Physicians** | Receive a progression risk score alongside routine lab results to guide treatment decisions |
| **Health Insurance Companies** | Risk-score diabetic policyholders for more accurate premium pricing and proactive care management |
| **Pharmaceutical Companies** | Stratify patients for clinical trials based on measurable disease burden |
| **Public Health Departments** | Forecast diabetes burden at a population level for resource planning |
| **Telehealth Platforms** | Trigger automated alerts for remote patients whose predicted score is high |

---

## 🌐 Live Application

The app is deployed on **Streamlit Community Cloud** and accessible at:

👉 [https://diabetes-prediction-by-subhamsingh.streamlit.app/](https://diabetes-prediction-by-subhamsingh.streamlit.app/)

---

## 👤 Author

**Subham Singh** — [github.com/subxm](https://github.com/subxm)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).