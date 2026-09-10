# FUNAAB ICTREC Predictive Helpdesk Triage

![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-blue?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn&logoColor=white)

An interactive, AI-driven decision support system designed for the **FUNAAB ICTREC Helpdesk**. This application uses Natural Language Processing (NLP) and Cost-Sensitive Machine Learning to instantly predict if a newly submitted helpdesk ticket will breach Service Level Agreements (SLAs), allowing for proactive "Minute-1 Escalation."

## 🚀 The Problem & Solution
Traditional helpdesk systems route tickets based on static, user-selected fields (like "Priority" or "Category"). Our research on historical ICTREC instances revealed that these static fields only predict resolution time with **~55% accuracy**. A user might mark an issue as "Low Priority," but the actual underlying complexity could take days to resolve.

**The Solution:**
Instead of waiting 8 hours to realize a ticket is complex, this application utilizes a **Hybrid NLP + XGBoost Pipeline**. It analyzes the actual terminology in the user's `Issue_Description` combined with categorical metadata. It calculates a real-time probability of delay and uses Cost-Sensitive Learning to trigger an immediate escalation to Senior Technicians.

## 🧠 Architecture
This project is decoupled into production-ready layers:
*   `train_and_save.py`: The model pipeline builder. Uses `TfidfVectorizer` to extract text features, merges them with One-Hot Encoded categorical variables, and trains an `XGBClassifier`. The model uses custom class weighting to heavily penalize false negatives (missed delayed tickets) to protect SLAs.
*   `predictor.py`: The inference service layer. Handles model serialization and provides a lightweight Explainable AI (XAI) feature to map high-risk vocabulary back to the user's text.
*   `app.py`: The interactive Streamlit frontend. Provides the "Live Triage Simulator" and the "Executive Impact Dashboard".

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
3. (Optional) Re-train the model if needed:
   ```bash
   python train_and_save.py
   ```
4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open your browser to `http://localhost:8501`.

## 📊 Presentation Features
This application is built for live demonstration to judging panels:
*   **Live Simulator**: Type in mock issues and watch the probability gauge react in real-time.
*   **One-Click Presets**: Use the "Baseline" and "Deceptive Ticket" buttons for flawless live pitches without typing.
*   **Explainable AI**: The app explicitly highlights the exact words (e.g., "portal", "gateway", "down") that triggered the model's alert.
*   **Executive Impact Calculator**: Translates raw machine learning metrics into projected business value (e.g., "Net Operational Hours Saved" per 1,000 tickets).

## 📄 Repository Structure
*   `/app_data/` - Contains the serialized `.pkl` pipeline and exported evaluation CSVs.
*   `app.py` - Main Streamlit UI.
*   `predictor.py` - Decoupled inference engine.
*   `train_and_save.py` - Model training script.
*   `requirements.txt` - Deployment dependencies for Streamlit Cloud.
*   `FUNAAB_Helpdesk_Analysis.ipynb` - The original Jupyter Notebook analysis.
