import pandas as pd
from elasticsearch import Elasticsearch

es_model_id = 'intfloat__multilingual-e5-base'

from elasticsearch import Elasticsearch

url = "https://b675458383584374af04fcee51788385.us-central1.gcp.cloud.es.io"
api_id = "VR-wlpoBpE2dlQuNl5eB"
api_key = "Bj6KJAUz1Yha5y6CgeYtrQ"

client = Elasticsearch(
    url,
    api_key=(api_id, api_key)
)

def internal_search(query):

    response = client.search(
        index="class-info",
        knn={
            "field": "chunk_embedding.predicted_value",
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": es_model_id,
                    "model_text": f"query: {query}"
                }
            },
            "k": 5,
            "num_candidates": 20,
        }
    )

    formatted_results = []
    for hit in response["hits"]["hits"]:
        result = {
            "score": hit["_score"],
            "file_name": hit["_source"]["filename"],
            "chunk_no": hit["_source"]["chunk_seq"],
            "chunk_text": hit["_source"]["chunk_text"]
        }
        formatted_results.append(result)

    return formatted_results

print(internal_search("tf-idf와 bm25의 차이는?"))