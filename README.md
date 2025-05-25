# FakeSense API

A FastAPI-based web service for detecting deepfake videos using a multimodal deep learning model.

## API Endpoints

1. `GET /`: Welcome message
2. `POST /predict`: Upload a video file for deepfake detection

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn app:app --reload
```

3. Access the API documentation at `http://localhost:8000/docs`

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## API Usage

### Using curl:
```bash
curl -X POST "https://your-render-url/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/video.mp4"
```

### Using Python requests:
```python
import requests

url = "https://your-render-url/predict"
files = {"file": open("path/to/your/video.mp4", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Response Format

```json
{
    "status": "success",
    "prediction": "Real/Fake",
    "confidence": "XX.XX%"
}
```

## Error Response

```json
{
    "status": "error",
    "message": "Error description"
}
``` 