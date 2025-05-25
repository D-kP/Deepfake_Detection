from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from inference import predict_video

app = FastAPI(title="FakeSense API", description="API for detecting deepfake videos")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to FakeSense API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Make prediction
        result = predict_video(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)

        if result is not None:
            label_map = {0: "Real", 1: "Fake"}
            return {
                "status": "success",
                "prediction": label_map.get(result["predicted_label"], "Unknown"),
                "confidence": f"{result['confidence']*100:.2f}%",
                "frames": result["frames"],
                "frames_analyzed": result["frames_analyzed"],
                "processing_time": f"{result['processing_time']:.2f}s",
                "model_confidence": f"{result['model_confidence']*100:.2f}%"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to process video"
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 