from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from models.schemas import MessageResponse, HealthResponse

# create fastapi app instance
app = FastAPI(
    title="TranscriPy",
    description="transcription service api",
    version="1.0.0"
)

# add cors middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)

# include api router
app.include_router(api_router)


@app.get("/", response_model=MessageResponse)
async def root():
    """root endpoint to check if api is working"""
    return {"message": "transcripy api is running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """health check endpoint"""
    return {"status": "healthy"}

