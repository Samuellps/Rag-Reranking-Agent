"""
Confgurações no Agente Orquestrador
- USE_THRESHOLD: Se True, ativa a filtragem de similaridade.
- SIMILARITY_THRESHOLD: Limite de similaridade para considerar um resultado relevante.
- USE_RERANK: Se True, ativa o reranking dos resultados.
- USE_HYDE: Se True, ativa o uso do modelo Hyde para gerar respostas hipotéticas.
"""

# --- Configurações Rerank Cohere ---
USE_RERANK = False

# --- Configurações de filtragem ---
USE_THRESHOLD = True
SIMILARITY_THRESHOLD = 0.2

# --- Configurações do Hyde RAG
USE_HYDE = True


# --- Maxímo de tokens por chunk ---
CHUNK_SIZE = 500



