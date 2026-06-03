# Next implementation steps for MindLink

## Phase 1: Public dataset prototype

1. Download WESAD.
2. Train the Random Forest model.
3. Report subject-independent F1 score.
4. Save the model.
5. Build a simple dashboard.

## Phase 2: Personal baseline model

1. Collect 3-7 days of normal user data.
2. Calculate each user's personal baseline.
3. Detect deviations from that baseline.
4. Add caregiver feedback labels.

## Phase 3: Your own device

1. Pick sensors:
   - PPG for pulse/HR/HRV
   - EDA for skin conductance
   - temperature sensor
   - accelerometer

2. Stream sensor data to phone/computer.
3. Convert raw data into the same features.
4. Retrain the model on your own data.

## Phase 4: Better AI

After Random Forest works, try:

1. XGBoost
2. 1D CNN
3. LSTM
4. Transformer time-series model

Do not start with deep learning. Start simple and prove the pipeline first.
