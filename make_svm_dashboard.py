
"""
Create an SVM version of your existing dashboard.
Run inside mindlink_ai_starter:
    python make_svm_dashboard.py
Then run:
    streamlit run mindlink_dashboard_v2_svm.py
"""
from pathlib import Path

source = Path('mindlink_dashboard_v2.py')
if not source.exists():
    raise FileNotFoundError('mindlink_dashboard_v2.py not found in this folder.')

text = source.read_text(encoding='utf-8')
text = text.replace('models/mindlink_random_forest.joblib', 'models/mindlink_svm_rbf.joblib')
text = text.replace('Random Forest', 'SVM RBF')
text = text.replace('random forest', 'SVM RBF')
text = text.replace('MindLink Personalized Baseline Dashboard v2', 'MindLink Personalized Baseline Dashboard v2 — SVM RBF')

out = Path('mindlink_dashboard_v2_svm.py')
out.write_text(text, encoding='utf-8')
print('Created:', out)
