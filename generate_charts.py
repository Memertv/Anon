import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create assets folder
if not os.path.exists('assets'):
    os.makedirs('assets')

# Baseline CM (from Random Forest run)
# Actual Fast (0): 54, Actual Medium (1): 46
# Fast: 32 (TP for Fast), 22 (FN for Fast)
# Medium: 19 (FN for Medium), 27 (TP for Medium)
cm_base = [[32, 22], [19, 27]]

# Advanced CM (loaded from app_data/confusion_matrix.csv if available)
cm_adv_df = pd.read_csv('app_data/confusion_matrix.csv', index_col=0)
cm_adv = cm_adv_df.values

def plot_cm(cm, title, filename):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Fast (< 8 hrs)', 'Pred Delayed (> 8 hrs)'],
                yticklabels=['Actual Fast (< 8 hrs)', 'Actual Delayed (> 8 hrs)'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'assets/{filename}')
    plt.close()

plot_cm(cm_base, 'Baseline Model (Random Forest) Confusion Matrix', 'baseline_cm.png')
plot_cm(cm_adv, 'Advanced Model (XGBoost + NLP) Confusion Matrix', 'advanced_cm.png')

print("Charts generated in assets/ folder.")
