import os
from dotenv import load_dotenv
from rag_graph import EnhancedRAGGraph

load_dotenv()

def main():
    # Google Custom Search API 설정 (선택사항)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    # Enhanced RAG 시스템 초기화
    rag = EnhancedRAGGraph(google_api_key, google_search_engine_id)
    
    print("이 시스템은 질문에 따라 내부 검색/Google 검색/arXiv 논문 검색을 지능적으로 선택합니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    while True:
        try:
            question = input("질문을 입력하세요: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("시스템을 종료합니다.")
                break
            
            if not question:
                continue
            
            print("\n🔍 검색 중...")
            result = rag.ask(question)
            
            print(f"\n📝 질문: {result['question']}")
            print(f"🔧 최적화된 쿼리: {result['optimized_query']}")
            print(f"🎯 선택된 검색 방법: {result['search_route']}")
            
            print(f"\n📊 검색 결과 요약:")
            print(f"  - 내부 문서: {result['internal_source_count']}개")
            print(f"  - Google 검색: {result['google_source_count']}개")
            print(f"  - arXiv 논문: {result['arxiv_source_count']}개")
            
            print(f"\n💡 답변:")
            print(result['answer'])
            
            # 상세 정보 표시 옵션
            show_details = input("\n상세 검색 결과를 보시겠습니까? (y/n): ").strip().lower()
            if show_details == 'y':
                print("\n" + "="*50)
                print("상세 검색 결과")
                print("="*50)
                
                # 내부 문서 결과
                if result['retrieved_docs']:
                    print("\n📁 내부 문서 검색 결과:")
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        print(f"\n[{i}] 파일: {doc['file_name']}")
                        print(f"    점수: {doc['score']:.3f}")
                        print(f"    내용: {doc['chunk_text'][:200]}...")
                
                # Google 검색 결과
                google_results = result.get('google_results', [])
                if google_results:
                    print(f"\n🌐 웹 검색 결과:")
                    for i, res in enumerate(google_results, 1):
                        print(f"\n[{i}] {res['title']}")
                        print(f"    URL: {res['url']}")
                        print(f"    내용: {res['snippet']}")
                
                # arXiv 검색 결과
                arxiv_results = result.get('arxiv_results', [])
                if arxiv_results:
                    print(f"\n📚 논문 검색 결과:")
                    for i, paper in enumerate(arxiv_results, 1):
                        authors = ", ".join(paper['authors'][:3])
                        if len(paper['authors']) > 3:
                            authors += " 외"
                        
                        print(f"\n[{i}] {paper['title']}")
                        print(f"    저자: {authors}")
                        print(f"    발행일: {paper['published_date']}")
                        print(f"    PDF: {paper['pdf_url']}")
                        print(f"    요약: {paper['summary'][:150]}...")
            
            print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()