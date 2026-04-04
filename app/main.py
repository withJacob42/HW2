from fastapi import FastAPI, File, UploadFile, HTTPException
from app.model import classifier

app = FastAPI(
    title="Food Image Classification API",
    description="An API serving a lightweight model for food image classification. Intended for MLOps pipelines.",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {
        "status": "Healthy", 
        "message": "Welcome to the Food Classification API. Use POST /predict to classify images."
    }

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
            "confidence": float(prediction["confidence"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
