from typing import List, Dict, Annotated

from fastapi import APIRouter, Path, Body
from fastapi import Depends, HTTPException

from pydantic import BaseModel
from uuid import uuid4

from ..models import User
from .users import get_current_active_user


from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore

from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQA


embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004"
)

bq_store = BigQueryVectorStore(
    project_id="dataton-2024-team-05-cofares",
    dataset_name="datahub_marina",
    table_name="O_T_articulos_datathon_cofares",
    location="eu",
    embedding=embedding_model,
)

llm = VertexAI(model_name="gemini-pro")
retriever = bq_store.as_retriever()
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)

router = APIRouter()

class ChatMessage(BaseModel):
    user: str
    message: str

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage] = []

# In-memory storage for chat sessions (replace with database in production)
chat_sessions: Dict[str, ChatSession] = {}

@router.post("/chat/start")
def start_chat(current_user: User = Depends(get_current_active_user)):
    """Starts a new chat session for the user."""
    session_id = str(uuid4())
    chat_sessions[session_id] = ChatSession(session_id=session_id)
    return {"session_id": session_id}

@router.post("/chat/{session_id}/send")
def send_message(session_id: Annotated[str, Path(title="The ID of the item to get")],
                 message: str = Body(..., embed=True),
                 current_user: User = Depends(get_current_active_user)
                 ):
    """Sends a message to the specified chat session."""
    # session = chat_sessions.get(session_id)
    # if not session:
    #     raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    print(message)
    
    # session.messages.append(ChatMessage(user=current_user.username, message=message))

    # Here you would call your chatbot logic and get the response
    chatbot_response = get_chatbot_response(message) 

    # session.messages.append(ChatMessage(user="chatbot", message=chatbot_response))
    return {"response": chatbot_response}

@router.get("/chat/{session_id}/history")
def get_chat_history(session_id: str, current_user: User = Depends(get_current_active_user)):
    """Retrieves the message history for the specified chat session."""
    session = chat_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    return session.messages

# Placeholder function for chatbot logic
def get_chatbot_response(search_query: str) -> str:
    try:
        res = retrieval_qa.invoke(search_query)
    except Exception as e:
        print(str(e))
        return "No se pudo obtener una respuesta"
    return res
