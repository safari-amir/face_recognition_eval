from fastapi import FastAPI, File, UploadFile, HTTPException
from face_rec import FaceMatcher  # Import the FaceMatcher class
from typing import Optional

app = FastAPI(title="Face Recognition API")

# Global instance of FaceMatcher
face_matcher: Optional[FaceMatcher] = None

@app.post("/initialize")
def initialize(model_name: str):
    """
    Initialize the FaceMatcher model.
    
    Args:
        model_name (str): The name of the face recognition model (e.g., "buffalo_s").
    
    Returns:
        dict: Confirmation message.
    """
    global face_matcher
    face_matcher = FaceMatcher(model_name=model_name)
    return {"message": "Model initialized successfully."}

@app.post("/predict")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compare two face images and determine if they match.
    
    Args:
        file1 (UploadFile): The first uploaded image.
        file2 (UploadFile): The second uploaded image.
    
    Returns:
        dict: {"result": 1} if faces match, {"result": 0} otherwise.
    """
    if face_matcher is None:
        raise HTTPException(status_code=400, detail="Model is not initialized. Call /initialize first.")

    try:
        # Read the images
        img1 = await file1.read()
        img2 = await file2.read()

        # Perform face matching
        result = face_matcher.match_face(img1, img2)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in one or both images.")

        return {"result": 1 if result else 0}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
