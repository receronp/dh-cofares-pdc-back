import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore

# Initialize the embedding model
embedding_model = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002"
)

# Create the BigQueryVectorStore
bq_store = BigQueryVectorStore(
    project_id="dataton-2024-team-05-cofares",
    dataset_name="datahub_marina",
    table_name="embeddings_multilingual_LIMIT_10000",
    location="eu",
    embedding=embedding_model,
)

# Define the query
query = "Qué medicamento ayuda a controlar la presión arterial?"

# Perform semantic search
results = bq_store.similarity_search(query, k=5)

# Process the results
results = [json.loads(x.model_dump_json()) for x in results]
print(json.dumps(results, indent=2))
