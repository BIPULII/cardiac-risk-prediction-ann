Run these commands from the project folder:

cd "/Users/bipuli/Academic/AI project - 6 th sem/cardiac-risk-prediction-ann"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

1) Train the model

PYTHONPATH=. python -m src.train_model


2) Start the backend

PYTHONPATH=. python -m uvicorn src.api:app --host 127.0.0.1 --port 8080

3) Start the frontend

python -m streamlit run app.py --server.headless true --server.port 8501

Then open:

Frontend: http://localhost:8501
API: http://127.0.0.1:8080/docs
The trained model is already saved in heart_ann_model.keras, so you can demo it immediately after starting the backend and UI.





Here are the steps:

Option 1: Run Training with Full Evaluation
Generates:
python3 src/train_model.py

outputs/model_evaluation_metrics.csv - All 11 metrics
outputs/training_evaluation.png - 4-panel chart (accuracy, loss, confusion matrix, ROC-AUC)
heart_ann_model.keras - Trained model

Option 2: Compare All Models
Generates:
python3 src/compare_models.py

model_comparison.csv - Side-by-side comparison of Logistic Regression, Random Forest, and your ANN
outputs/comprehensive_model_comparison.png - 4-panel comparison chart

Quick Start (Run both):
python3 src/train_model.py && python3 src/compare_models.py

View the Results:
cat outputs/model_evaluation_metrics.csv
cat outputs/model_comparison.csv

CSV Files (spreadsheet format):

PNG Images (charts):

Open outputs/training_evaluation.png
Open outputs/comprehensive_model_comparison.png
All outputs are saved in the outputs folder automatically. The PNG files contain all your evaluation visualizations.

for installed the required packages 
pip3 install -r requirements.txt