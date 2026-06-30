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