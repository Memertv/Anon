import pandas as pd
import joblib
import numpy as np

# Load model globally so it's cached in memory
MODEL_PATH = "app_data/triage_model.pkl"

def load_pipeline():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None

def predict_ticket(pipeline, user_type, unit_dept, issue_category, priority, issue_desc):
    """
    Takes raw inputs from the UI, formats them into a DataFrame, and returns the probability of Delay.
    """
    input_data = pd.DataFrame([{
        'User_Type': user_type,
        'Unit_Dept': unit_dept,
        'Issue_Category': issue_category,
        'Priority': priority,
        'Issue_Description': issue_desc
    }])
    
    # Predict Probability of Class 1 (Medium/Delayed)
    prob = pipeline.predict_proba(input_data)[0][1]
    return float(prob)

def get_high_risk_words(pipeline):
    """
    Lightweight Explainable AI (XAI).
    Extracts the TF-IDF vocabulary and maps it to the XGBoost feature importances
    to find the words most associated with predicting a delay.
    """
    try:
        # Extract components from pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        # Get TF-IDF vectorizer and its feature names
        tfidf = preprocessor.transformers_[0][1]
        text_features = tfidf.get_feature_names_out()
        
        # The text features are the first N features in the XGBoost model
        num_text_features = len(text_features)
        
        # Get importances for just the text features
        text_importances = classifier.feature_importances_[:num_text_features]
        
        # Create a mapping of word -> importance
        word_importances = list(zip(text_features, text_importances))
        
        # Sort by importance
        word_importances.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 20 words
        return [word for word, imp in word_importances[:20] if imp > 0]
    except Exception as e:
        return []

def explain_prediction(pipeline, text):
    """
    Checks if any high-risk words are in the provided text.
    """
    high_risk_words = get_high_risk_words(pipeline)
    text_lower = text.lower()
    found_words = [word for word in high_risk_words if word in text_lower]
    return found_words
