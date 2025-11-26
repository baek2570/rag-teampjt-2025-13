import os
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

url = os.getenv("ELASTIC_CLOUD_URL")
api_id = os.getenv("ELASTIC_API_ID")
api_key = os.getenv("ELASTIC_API_KEY")
es_model_id = os.getenv("ELASTIC_MODEL_ID")

client = Elasticsearch(
    url,
    api_key=(api_id, api_key)
)

def internal_search(query, top_k=5):

    search_kwargs = {
        "index": "class-info",
        
        "retriever": {
            "rrf": { 
                "rank_window_size": 50,
                "rank_constant": 60,
                "retrievers": [
                    # 1) kNN 검색 (DenseSearch)
                    {
                        "knn": {
                            "field": "chunk_embedding.predicted_value",
                            "k": top_k,
                            "num_candidates": 20,
                            "query_vector_builder": {
                                "text_embedding": {
                                    "model_id": es_model_id,
                                    "model_text": f"query: {query}"
                                }
                            }
                        }
                    },
                    # 2) BM25 검색 (SparseSearch)
                    {
                        "standard": {
                            "query": {
                                "match": {
                                    "chunk_text": {
                                        "query": query
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        },
        
        "size": top_k,
    }

    search_kwargs["_source"] = {"exclude_vectors": True}
    response = client.search(**search_kwargs)

    formatted_results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        result = {
            "score": hit["_score"],        
            "file_name": src.get("filename"),
            "chunk_no": src.get("chunk_seq"),
            "chunk_text": src.get("chunk_text"),
        }
        formatted_results.append(result)

    return formatted_results

print(internal_search("tf-idf와 bm25의 차이는?", 5))