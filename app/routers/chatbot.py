import json
from typing import Annotated, List, Dict, Any

from fastapi import APIRouter, Path, Body
from fastapi import Depends, HTTPException

from pydantic import BaseModel
from uuid import uuid4

from ..models import User, ChatRequest
from .users import get_current_active_user

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore


# Initialize the embedding model
embedding_model = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
llm = ChatVertexAI(model="gemini-1.5-flash")

# Create the BigQueryVectorStore
bq_store = None
system_prompt = (
    "Eres un asistente farmaceutico para tareas de recomendación de productos. "
    "Utiliza los siguientes elementos del contexto recuperado para responder a la pregunta. "
    "Si no conoce la respuesta, di que no lo sabes, no intentes inventar una respuesta. "
    "Utiliza tres frases como máximo y procura que la respuesta sea concisa."
    "\n\n"
    "{context}"
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


# @router.post("/chat/start")
def start_chat(current_user: User = Depends(get_current_active_user)):
    """Starts a new chat session for the user."""
    session_id = str(uuid4())
    chat_sessions[session_id] = ChatSession(session_id=session_id)
    return {"session_id": session_id}


# @router.post("/chat/{session_id}/send")
@router.post("/chat/send")
def send_message(
    # session_id: Annotated[str, Path(title="The ID of the item to get")],
    chat_request: ChatRequest,
    # current_user: User = Depends(get_current_active_user),
):
    """Sends a message to the specified chat session."""
    # session = chat_sessions.get(session_id)
    # if not session:
    #     raise HTTPException(status_code=404, detail="Sesión no encontrada")

    # session.messages.append(ChatMessage(user=current_user.username, message=message))

    global bq_store
    bq_store = BigQueryVectorStore(
        project_id="dataton-2024-team-05-cofares",
        dataset_name="datahub_marina",
        table_name=chat_request.table_name,
        location="eu",
        embedding=embedding_model,
    )

    retriever = bq_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": chat_request.message})

    # Here you would call your chatbot logic and get the response
    chatbot_response = get_chatbot_response(chat_request.message, chat_request.k)

    # session.messages.append(ChatMessage(user="chatbot", message=response["answer"]))
    return {"response": response["answer"], "recomendaciones": chatbot_response}


# @router.get("/chat/{session_id}/history")
def get_chat_history(
    session_id: str, current_user: User = Depends(get_current_active_user)
):
    """Retrieves the message history for the specified chat session."""
    session = chat_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    return session.messages


# Placeholder function for chatbot logic
def get_chatbot_response(search_query: str, k: int) -> List[Any]:
    try:
        # Perform semantic search
        results = bq_store.similarity_search(search_query, k)

        # Process the results
        results = [json.loads(x.model_dump_json()) for x in results]
        return results
    except Exception as e:
        return [{"error": str(e)}]
