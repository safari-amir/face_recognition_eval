from fastapi import FastAPI, File, UploadFile, HTTPException
from face_rec import FaceMatcher  # Import the consolidated class

app = FastAPI(title="Face Recognition")
face_matcher = None  # Global instance of FaceMatcher

@app.post("/initialize")
def initialize(model_name: str):
    global face_matcher
    face_matcher = FaceMatcher(model_name=model_name)
    return {"message": "Model initialized"}

@app.post("/predict")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if face_matcher is None:
        raise HTTPException(status_code=400, detail="Model is not initialized.")
    
    img1 = await file1.read()
    img2 = await file2.read()
    
    # Use the match_face method of FaceMatcher
    result = face_matcher.match_face(img1, img2)
    return {"result": 1 if result else 0}
