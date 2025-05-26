from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import uvicorn
import os
import joblib  # Thêm thư viện để lưu/load model
from datetime import datetime

# Khởi tạo FastAPI app
app = FastAPI(
    title="Student Dropout Risk Prediction API",
    description="API để dự đoán nguy cơ bỏ học của sinh viên",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Models cho request/response
class Student(BaseModel):
    email: str
    name: str
    days_since_enrollment: int
    passedLesson: int
    passedTests: int
    progress: float
    lastLogin: str


class StudentWithRisk(BaseModel):
    email: str
    name: str
    days_since_enrollment: int
    passedLesson: int
    passedTests: int
    progress: float
    lastLogin: str
    risk: int
    risk_status: str


class PredictionRequest(BaseModel):
    students: List[Student]


class PredictionResponse(BaseModel):
    predictions: List[StudentWithRisk]
    summary: dict


# Global model variable
model = None
MODEL_FILE = 'student_dropout_model.pkl'  # File để lưu model
TRAINING_DATA_FILE = 'student_training_data.csv'


def train_and_save_model():
    """Train model từ CSV và lưu vào file"""
    try:
        # Kiểm tra file CSV có tồn tại không
        if not os.path.exists(TRAINING_DATA_FILE):
            raise FileNotFoundError(f"Không tìm thấy file {TRAINING_DATA_FILE}")

        # Đọc dữ liệu từ CSV
        print(f"Đang đọc dữ liệu từ {TRAINING_DATA_FILE}...")
        df = pd.read_csv(TRAINING_DATA_FILE)

        # Kiểm tra các cột cần thiết
        required_columns = ['days_since_enrollment', 'passedLesson', 'passedTests', 'progress', 'risk']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Thiếu các cột: {missing_columns}")

        # Tách features và labels
        X_train = df[['days_since_enrollment', 'passedLesson', 'passedTests', 'progress']].values
        y_train = df['risk'].values

        print(f"Đã load {len(df)} mẫu dữ liệu training")
        print(f"Số lượng sinh viên có nguy cơ bỏ học: {sum(y_train)}")
        print(f"Số lượng sinh viên an toàn: {len(y_train) - sum(y_train)}")

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Lưu model vào file
        joblib.dump(model, MODEL_FILE)
        print(f"✓ Model đã được train và lưu vào {MODEL_FILE}")

        # Lưu thông tin metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(df),
            'high_risk_count': int(sum(y_train)),
            'safe_count': int(len(y_train) - sum(y_train))
        }

        import json
        with open('model_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return model

    except Exception as e:
        print(f"Lỗi khi train model: {e}")
        raise e


def load_saved_model():
    """Load model đã lưu từ file"""
    try:
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            print(f"✓ Đã load model từ {MODEL_FILE}")

            # Hiển thị thông tin metadata nếu có
            if os.path.exists('model_metadata.json'):
                import json
                with open('model_metadata.json', 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"  - Được train lúc: {metadata.get('trained_at', 'N/A')}")
                print(f"  - Số mẫu training: {metadata.get('training_samples', 'N/A')}")
                print(f"  - Sinh viên nguy cơ cao: {metadata.get('high_risk_count', 'N/A')}")
                print(f"  - Sinh viên an toàn: {metadata.get('safe_count', 'N/A')}")

            return model
        else:
            print(f"Không tìm thấy file model {MODEL_FILE}")
            return None
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return None


def load_or_train_model():
    """Load model từ file, nếu không có thì train mới"""
    global model

    # Thử load model đã lưu trước
    model = load_saved_model()

    if model is None:
        print("Không có model đã lưu, bắt đầu train model mới...")
        model = train_and_save_model()

    return model


@app.on_event("startup")
async def startup_event():
    """Load hoặc train model khi start server"""
    try:
        load_or_train_model()
        print("✓ Model sẵn sàng để dự đoán!")
    except Exception as e:
        print(f"✗ Lỗi khi load/train model: {e}")
        print("Server sẽ không thể dự đoán cho đến khi model được load thành công")


@app.get("/")
async def root():
    return {
        "message": "Student Dropout Risk Prediction API",
        "status": "ready",
        "endpoints": {
            "predict": "/predict - Dự đoán nguy cơ bỏ học",
            "predict-simple": "/predict-simple - API đơn giản",
            "retrain": "/retrain - Train lại model từ CSV",
            "model-info": "/model-info - Thông tin model",
            "health": "/health - Kiểm tra trạng thái",
            "docs": "/docs - Tài liệu API"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file_exists": os.path.exists(MODEL_FILE),
        "training_data_exists": os.path.exists(TRAINING_DATA_FILE),
        "ready_for_predictions": model is not None
    }


@app.get("/model-info")
async def get_model_info():
    """Lấy thông tin về model hiện tại"""
    info = {
        "model_loaded": model is not None,
        "model_file_exists": os.path.exists(MODEL_FILE),
        "training_data_exists": os.path.exists(TRAINING_DATA_FILE)
    }

    # Thêm metadata nếu có
    if os.path.exists('model_metadata.json'):
        import json
        try:
            with open('model_metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            info["metadata"] = metadata
        except:
            pass

    return info


@app.post("/retrain")
async def retrain_model():
    """Train lại model từ CSV và lưu vào file"""
    global model

    try:
        model = train_and_save_model()
        return {
            "message": "Model đã được train lại thành công",
            "status": "success",
            "model_file": MODEL_FILE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi train model: {str(e)}")


# OPTIONS handler cho preflight requests
@app.options("/predict-simple")
async def options_predict_simple():
    return {"message": "OK"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_dropout_risk(request: PredictionRequest):
    """Dự đoán nguy cơ bỏ học cho danh sách sinh viên"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")

    if not request.students:
        raise HTTPException(status_code=400, detail="Danh sách sinh viên không được rỗng")

    try:
        # Chuẩn bị dữ liệu cho ML model
        X = np.array([[s.days_since_enrollment, s.passedLesson, s.passedTests, s.progress]
                      for s in request.students])

        # Dự đoán
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Tạo kết quả
        results = []
        for i, (student, pred, prob) in enumerate(zip(request.students, predictions, probabilities)):
            risk_status = "NGUY CƠ BỎ HỌC" if pred == 1 else "AN TOÀN"

            result = StudentWithRisk(
                email=student.email,
                name=student.name,
                days_since_enrollment=student.days_since_enrollment,
                passedLesson=student.passedLesson,
                passedTests=student.passedTests,
                progress=student.progress,
                lastLogin=student.lastLogin,
                risk=int(pred),
                risk_status=risk_status
            )
            results.append(result)

        # Tạo summary
        total_students = len(results)
        high_risk_count = sum(1 for r in results if r.risk == 1)
        safe_count = total_students - high_risk_count

        summary = {
            "total_students": total_students,
            "high_risk_students": high_risk_count,
            "safe_students": safe_count,
            "high_risk_percentage": round((high_risk_count / total_students) * 100, 1) if total_students > 0 else 0
        }

        return PredictionResponse(predictions=results, summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")


@app.post("/predict-simple")
async def predict_simple(students: List[List[float]]):
    """API đơn giản - Input: [[days, lessons, tests, progress], ...]"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")

    if not students:
        raise HTTPException(status_code=400, detail="Danh sách sinh viên không được rỗng")

    try:
        X = np.array(students)
        predictions = model.predict(X)

        results = []
        for i, (student, pred) in enumerate(zip(students, predictions)):
            result = student + [int(pred)]
            results.append(result)

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )