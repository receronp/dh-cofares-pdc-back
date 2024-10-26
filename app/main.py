from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from .routers import users

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)

@app.get("/")
async def root():
    return {"message": "Bienvenido a Cofares AI"}

@app.get("/health")
async def health():
    return {"status": "ok"}
