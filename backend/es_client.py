from elasticsearch import Elasticsearch

url = "https://b675458383584374af04fcee51788385.us-central1.gcp.cloud.es.io"
api_id = "VR-wlpoBpE2dlQuNl5eB"
api_key = "Bj6KJAUz1Yha5y6CgeYtrQ"
model_id = "intfloat__multilingual-e5-base"


class EsClient:
    client = Elasticsearch(
        url,
        api_key=(api_id, api_key)
    )

    def internal_search(self, query, top_k=5):
        search_kwargs = {
            "index": "class-info",
            "retriever": {
                "rrf": {
                    "rank_window_size": 50,
                    "rank_constant": 60,
                    "retrievers": [
                        {
                            "knn": {
                                "field": "chunk_embedding.predicted_value",
                                "k": top_k,
                                "num_candidates": 20,
                                "query_vector_builder": {
                                    "text_embedding": {
                                        "model_id": model_id,
                                        "model_text": f"query: {query}"
                                    }
                                }
                            }
                        },
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
            "_source": {"exclude_vectors": True}
        }
        response = self.client.search(**search_kwargs)

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
