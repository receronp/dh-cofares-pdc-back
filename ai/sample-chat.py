from langchain_google_vertexai import ChatVertexAI

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore

llm = ChatVertexAI(model="gemini-1.5-flash")

# Initialize the embedding model
embedding_model = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")

# Create the BigQueryVectorStore
bq_store = BigQueryVectorStore(
    project_id="dataton-2024-team-05-cofares",
    dataset_name="datahub_marina",
    table_name="embeddings_multilingual_LIMIT_10000",
    location="eu",
    embedding=embedding_model,
)

retriever = bq_store.as_retriever()


# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "Eres un asistente farmaceutico para tareas de recomendación de productos. "
    "Utiliza los siguientes elementos del contexto recuperado para responder a la pregunta. "
    "Si no conoce la respuesta, di que no lo sabes, no intentes inventar una respuesta. "
    "Utiliza tres frases como máximo y procura que la respuesta sea concisa."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke(
    {"input": "Qué medicamento ayuda a controlar la presión arterial?"}
)
print(response["answer"])
