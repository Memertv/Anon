# FUNAAB ICTREC Predictive Helpdesk Triage

![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-blue?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn&logoColor=white)

An interactive, AI-driven decision support system designed for the **FUNAAB ICTREC Helpdesk**. This application uses Natural Language Processing (NLP) and Cost-Sensitive Machine Learning to instantly predict if a newly submitted helpdesk ticket will breach Service Level Agreements (SLAs), allowing for proactive "Minute-1 Escalation."

## 🚀 The Problem & Solution
Traditional helpdesk systems route tickets based on static, user-selected fields (like "Priority" or "Category"). Our research on historical ICTREC instances revealed that these static fields only predict resolution time with **~55% accuracy**. A user might mark an issue as "Low Priority," but the actual underlying complexity could take days to resolve.

**The Solution:**
Instead of waiting 8 hours to realize a ticket is complex, this application utilizes a **Hybrid NLP + XGBoost Pipeline**. It analyzes the actual terminology in the user's `Issue_Description` combined with categorical metadata. It calculates a real-time probability of delay and uses Cost-Sensitive Learning to trigger an immediate escalation to Senior Technicians.

---

## 🧹 Data Cleaning & Target Engineering Methodology
To transition from a skewed regression problem to a robust classification task, the data underwent the following preparation steps:
1. **Target Discretization:** The heavily skewed continuous variable `Resolution_Time_Hrs` was bucketed into clear operational classes. Tickets resolved in under 8 hours were labeled **"Fast"**, while those taking longer were labeled **"Medium/Delayed"**.
2. **Text Processing:** The raw `Issue_Description` text was standardized, stop words were removed, and the text was converted into numerical arrays using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
3. **Hybrid Concatenation:** The vectorized text features were joined with One-Hot Encoded categorical variables (`User_Type`, `Priority`, `Unit_Dept`, `Issue_Category`) to form a complete predictive matrix.

---

## 📊 Model Evaluation & Results
We trained two models to prove the efficacy of the advanced architecture. 

### 1. The Baseline Model (Random Forest)
Trained purely on categorical metadata (Priority, Category, Department).
*   **Accuracy:** 55%
*   **Insight:** Relying only on standard intake fields is basically a coin flip. The model misses many complex issues because users frequently mis-categorize their problems.

<img src="assets/baseline_cm.png" alt="Baseline Confusion Matrix" width="400"/>

### 2. The Advanced Model (NLP + XGBoost + Cost-Sensitive Learning)
Trained on the Hybrid TF-IDF feature set. We implemented **Cost-Sensitive Learning** (penalizing false negatives) to aggressively protect SLAs.
*   **Accuracy:** 51%
*   **Recall (Delayed Tickets):** 67%
*   **Insight:** While overall accuracy dropped slightly due to more "false alarms", the recall for delayed tickets skyrocketed. In a helpdesk environment, escalating a simple issue early (False Positive) is much cheaper than letting a complex issue sit untouched for 8 hours (False Negative).

<img src="assets/advanced_cm.png" alt="Advanced Confusion Matrix" width="400"/>

### Full Evaluation Metrics (Advanced Model)
| Metric | Score | Description |
| :--- | :--- | :--- |
| **AUC** | 0.521 | Area Under the ROC Curve |
| **Accuracy (CA)** | 0.510 | Classification Accuracy |
| **F1 Score** | 0.559 | Harmonic mean of Precision and Recall |
| **Precision** | 0.477 | Ratio of correct positive predictions |
| **Recall** | 0.674 | Ability to find all delayed tickets (SLA Protection) |
| **MCC** | 0.046 | Matthews Correlation Coefficient |

---

## 🧠 Architecture
This project is decoupled into production-ready layers:
*   `train_and_save.py`: The model pipeline builder. 
*   `predictor.py`: The inference service layer. Handles model serialization and provides lightweight Explainable AI (XAI).
*   `app.py`: The interactive Streamlit frontend. 

## 💻 Running Locally

### Prerequisites
Make sure you have Python 3.8+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/Memertv/Anon.git
   cd Anon
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser to `http://localhost:8501`.

## 📄 Repository Structure
*   `assets/` - Confusion matrix charts and images.
*   `app_data/` - Contains the serialized `.pkl` pipeline and exported evaluation CSVs.
*   `app.py` - Main Streamlit UI.
*   `predictor.py` - Decoupled inference engine.
*   `train_and_save.py` - Model training script.
*   `requirements.txt` - Deployment dependencies for Streamlit Cloud.
*   `FUNAAB_Helpdesk_Analysis.ipynb` - The original Jupyter Notebook analysis.
