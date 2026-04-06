from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.model import classifier
import os

app = FastAPI(
    title="Food Image Classification API",
    description="An API serving a lightweight model for food image classification. Intended for MLOps pipelines.",
    version="1.0.0"
)

# Setup static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>UI is missing</h1><p>Please place index.html in the app/static folder.</p>"

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Validate that the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    try:
        # Read the image file into bytes
        contents = await file.read()
        
        # Run inference
        prediction = classifier.predict(contents)
        
        return {
            "filename": file.filename,
            "prediction": prediction["class"],
            "confidence": float(prediction["confidence"]),
            "calories": prediction.get("calories", "Unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
