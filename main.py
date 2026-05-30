from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

app = FastAPI(title="Digital Twin ICU API", description="Real-time Anomaly Detection for Patient Vitals")

# ==========================================
# 1. تعريف معمارية الموديل (لازم تكون نفس التدريب بالظبط)
# ==========================================
class DigitalTwinClassifier(nn.Module):
    def __init__(self, n_features=7, hidden_dim=64):
        super(DigitalTwinClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hidden_n, _) = self.lstm(x)
        return self.classifier(hidden_n[-1]).squeeze()

# ==========================================
# 2. تحميل الأوزان والـ Scaler
# ==========================================
device = torch.device('cpu') # على Render المجاني هنشتغل CPU
model = DigitalTwinClassifier()
model.load_state_dict(torch.load('digital_twin_lstm_24h_97acc.pth', map_location=device))
model.eval()

scaler = joblib.load('digital_twin_scaler.pkl')

# ذاكرة مؤقتة لتخزين قراءات المرضى (عشان نكون نافذة الـ 24 ساعة)
patients_history = {}

# ==========================================
# 3. واجهة استقبال البيانات (Data Schema)
# ==========================================
class PatientVitals(BaseModel):
    patient_id: str
    Heart_Rate: float
    Respiratory_Rate: float
    SpO2: float
    Systolic_BP: float
    Diastolic_BP: float
    Temperature_C: float
    MAP: float

# ==========================================
# 4. الـ Endpoint الرئيسي (رادار الطوارئ)
# ==========================================
@app.post("/predict")
def predict_vitals(vitals: PatientVitals):
    try:
        # 1. تحويل القراءات لمصفوفة
        features = [vitals.Heart_Rate, vitals.Respiratory_Rate, vitals.SpO2, 
                    vitals.Systolic_BP, vitals.Diastolic_BP, vitals.Temperature_C, vitals.MAP]
        
        features_array = np.array(features).reshape(1, -1)
        
        # 2. التشفير (Scaling)
        scaled_features = scaler.transform(features_array)[0]
        
        # 3. إدارة النافذة الزمنية (Sliding Window)
        pid = vitals.patient_id
        if pid not in patients_history:
            # لو مريض جديد، بنكرر قراءته 24 مرة كبداية
            patients_history[pid] = [scaled_features] * 24
        else:
            # لو مريض قديم، بنرمي أقدم قراءة وندخل الجديدة (تحديث لحظي)
            patients_history[pid].pop(0)
            patients_history[pid].append(scaled_features)
            
        # 4. تجهيز الـ Tensor للـ PyTorch
        seq_array = np.array(patients_history[pid]) # shape: (24, 7)
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0) # shape: (1, 24, 7)
        
        # 5. اتخاذ القرار (Inference)
        with torch.no_grad():
            risk_score = model(seq_tensor).item()
            
        is_danger = bool(risk_score >= 0.5)
        
        return {
            "patient_id": pid,
            "risk_score": round(risk_score, 4),
            "status": "UNSTABLE (DANGER)" if is_danger else "STABLE",
            "alert": is_danger
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Digital Twin API is running! Go to /docs to test it."}