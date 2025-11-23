import os
from rag_graph import RAGGraph

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Failed to get OPENAI_API_KEY from environment variables")
        return

    rag = RAGGraph()
    while True:
        # "tf-idf와 bm25의 차이는?",
        # "머신러닝에서 overfitting이란 무엇인가요?",
        # "딥러닝과 머신러닝의 차이점을 설명해주세요.",
        # "자연어처리에서 임베딩이 무엇인지 알려주세요."

        question = input("\n질문을 입력하세요: ").strip()
        if not question:
            print("질문을 입력해주세요.")
            continue

        # RAG 실행
        print(f"\n🔍 Question: {question}")
        print("⏳ 검색 및 답변 생성 중...")

        try:
            result = rag.ask(question)

            print("\n" + "-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])

            print(f"\n 검색된 문서 수: {result['source_count']}")

            if result["retrieved_docs"]:
                print("\n Reference Documents:")
                for i, doc in enumerate(result["retrieved_docs"], 1):
                    print(f"  {i}. {doc['file_name']} (Score: {doc['score']:.3f})")

            print("\n" + "=" * 60)

        except Exception as e:
            print(f"Failed to Response: {e}")


if __name__ == "__main__":
    main()
