import os
import json
import pickle
import cohere
from cohere import Client as CohereClient 
import numpy as np
import openai 
from tqdm import tqdm 
from dotenv import load_dotenv



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

co = cohere.Client("COHERE_API_KEY") 

class SimpleVectorDB:
    def __init__(self, name, api_key=None): 
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"data/{name}/vector_db.pkl"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True) 
        self.total_tokens_used = 0 
        self.total_cost = 0.0
        self.client = openai.OpenAI(api_key=api_key if api_key else openai_api_key)


    def load_data(self, json_data):
        if os.path.exists(self.db_path):
            self.load_db()
            return

        texts_for_embedding = []
        for item in json_data:
            combined_text = f"Conteúdo do Chunk:\n{item['chunk']}\n\nContexto Adjacente Gerado:\n{item['context']}"
            texts_for_embedding.append(combined_text)
        
        metadata = [
            {
                "chunk_content": item["chunk"],
                "context": item["context"],     
                "original_index": idx
            }
            for idx, item in enumerate(json_data)
        ]
        self._embed_and_store(texts_for_embedding, metadata)
        self.save_db()
        
    def _embed_and_store(self, texts, metadata):
        batch_size = 128 
        result_embeddings = []
        total_tokens_for_batch = 0
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processando chunks para embedding"):
            batch_texts = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch_texts,
                model="text-embedding-3-small" 
            )
            
            result_embeddings.extend([res.embedding for res in response.data])
            total_tokens_for_batch += response.usage.total_tokens

        self.embeddings = result_embeddings
        self.metadata = metadata
        self.total_tokens_used += total_tokens_for_batch
        self.total_cost += (total_tokens_for_batch / 1_000_000) * 0.02

    def search(self, query, k=10, similarity_threshold=0.5, use_rerank=False, rerank_top_n=1):
        query_key = json.dumps({"query": query, "model": "text-embedding-3-small"}) 
        if query_key in self.query_cache:
            query_embedding = self.query_cache[query_key]
        else:
            response = self.client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            self.query_cache[query_key] = query_embedding
            
            self.total_tokens_used += response.usage.total_tokens
            self.total_cost += (response.usage.total_tokens / 1_000_000) * 0.02

        if not self.embeddings:
            return []

        similarities = np.dot(np.array(self.embeddings), query_embedding)
        
        effective_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:effective_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] >= similarity_threshold:
                results.append({
                    "chunk": self.metadata[idx]["chunk_content"],
                    "context": self.metadata[idx]["context"],
                    "similarity": float(similarities[idx]),
                    "original_index": self.metadata[idx]["original_index"],
                    "rank_after_similarity_search": rank + 1
                })

        if use_rerank and len(results) > 1:
            docs_to_rerank = [r["chunk"] for r in results]
            
            cohere_client = CohereClient(os.getenv("COHERE_API_KEY"))
            if not os.getenv("COHERE_API_KEY"):
               
                return results 
            try:
                rerank_response = cohere_client.rerank(
                    query=query,
                    documents=docs_to_rerank,
                    top_n=min(rerank_top_n, len(docs_to_rerank)),
                    model='rerank-multilingual-v3.0', 
                    return_documents=True
                )
                
                final_results_reranked = []
                for reranked_doc_obj in rerank_response.results:
                    original_result_for_reranked_doc = next(
                        (res for res in results if res["chunk"] == reranked_doc_obj.document['text']),
                        None 
                    )
                    if original_result_for_reranked_doc:
                        original_result_for_reranked_doc["rerank_score"] = reranked_doc_obj.relevance_score
                        final_results_reranked.append(original_result_for_reranked_doc)
                
                return final_results_reranked 
            except cohere.CohereAPIError as e:
                return results 
            except Exception as e:
                return results 
        else:
            return results

    def save_db(self):
        query_cache_json_str = json.dumps({k: list(v) if isinstance(v, np.ndarray) else v for k, v in self.query_cache.items()})
        data = {
            "embeddings": [list(e) for e in self.embeddings], 
            "metadata": self.metadata,
            "query_cache": query_cache_json_str, 
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
        }
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)
        
    def load_db(self):
        if not os.path.exists(self.db_path):
            return 

        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        
        self.embeddings = [np.array(e) for e in data.get("embeddings", [])] 
        self.metadata = data.get("metadata", [])
        query_cache_json_str = data.get("query_cache", "{}")
        self.query_cache = {k: np.array(v) if isinstance(v, list) else v for k, v in json.loads(query_cache_json_str).items()}
        self.total_tokens_used = data.get("total_tokens_used", 0)
        self.total_cost = data.get("total_cost", 0.0)
        
    def validate_embeddings(self): 
        if len(self.embeddings) != len(self.metadata):
            print(f"AVISO: Inconsistência - {len(self.embeddings)} embeddings vs {len(self.metadata)} metadados")
        else:
            print(f"Validação OK: {len(self.embeddings)} chunks processados")
        
        if self.metadata: 
            unique_chunks = len({meta["chunk_content"] for meta in self.metadata})
            print(f"Chunks únicos (baseado em 'chunk_content'): {unique_chunks}/{len(self.metadata)}")
        else:
            print("Metadados vazios, não é possível validar chunks únicos.")