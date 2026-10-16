from fastapi import FastAPI

app = FastAPI(title="ClipMotionfy API")

@app.get("/")
def home():
    return {
        "app": "ClipMotionfy",
        "status": "online"
    }
