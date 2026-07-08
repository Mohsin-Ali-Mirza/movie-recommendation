# MLFlow Model Training and Inference

## Requirements
Create a `requirements.txt` file with the following dependencies:
```
fastapi
requests
mlflow
scikit-learn
pandas
aiohttp
asyncio
uvicorn
numpy
torch
```
Install dependencies using:
```
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Run the Middleware API (FastAPI)
Start the middleware service in a new terminal:
```
python middleware.py
```

### Step 2: Run the Backend Inference Service
Open another terminal and start the backend service:
```
python backend.py
```

### Step 3: Run Multiple Workers (Optional for Testing)
Simulate multiple training requests with:
```
python multiple_workers.py
```

### Step 4: Open MLFlow UI to Track Experiments
To monitor training experiments, open a new terminal and run:
```
mlflow ui --host 127.0.0.1 --port 5000
```
Then, visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Batch Retraining Process
1. **Request Handling:** Users send training requests via the `middleware.py` service.
2. **Incremental Training Counter:**
   - A global counter tracks the number of requests.
   - Once the counter reaches the threshold (`THRESHOLD = 10`), a background task is triggered for retraining.
3. **Model Training & Registration:**
   - The middleware updates the dataset dynamically.
   - A new model is trained and logged in MLFlow.
   - The new champion model is selected based on accuracy.
   - The previous champion becomes a challenger for backup.
4. **Model Switching:**
   - When a new champion is detected, the backend asynchronously loads it.
   - The inference service automatically switches to the new best model.

---

## Notes
Due to time constraints, the batch retraining functionality was not directly integrated into the main backend code, but a working implementation is available.

