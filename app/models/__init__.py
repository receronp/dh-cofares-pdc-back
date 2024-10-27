from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class ChatRequest(BaseModel):
    message: str = None
    table_name: str | None = "embeddings_multilingual_LIMIT_10000"
    k: int | None = 5
