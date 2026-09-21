from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the student 2 router
from student2_cv.router import router as student2_router

app = FastAPI(
    title="AI Agent Services",
    description="Python backend with isolated agent modules."
)

# Add CORS middleware so the ASP.NET Core backend can call this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the student routers
app.include_router(student2_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Agent Backend!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
