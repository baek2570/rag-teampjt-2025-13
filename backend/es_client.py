from elasticsearch import Elasticsearch

url = "https://a30a7c8ef0bf435a9d350006622225d8.us-central1.gcp.cloud.es.io"
api_id = "7eFayZoBmaII0WM5wUsL"
api_key = "OfRBOxIfnESpqgMSAP39Jw"
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

        return formatted_results[:top_k]
