from typing import List, Dict, Any, TypedDict
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from es_client import EsClient
from search_tools import SearchToolsManager

load_dotenv()

class EnhancedRAGState(TypedDict):
    question: str
    optimized_query: str
    search_route: str
    retrieved_docs: List[Dict[str, Any]]
    google_results: List[Dict[str, Any]]
    arxiv_results: List[Dict[str, Any]]
    context: str
    answer: str
    messages: List[Any]

class EnhancedRAGGraph:
    def __init__(self, google_api_key: str = None, google_search_engine_id: str = None):
        self.es_client = EsClient()
        self.search_manager = SearchToolsManager(google_api_key, google_search_engine_id)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_PROXY_URL")
        )
        
        self.retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 질문을 분석하여 검색에 최적화된 쿼리를 생성하는 전문가입니다.
            사용자의 질문을 받아서 검색할 수 있는 효과적인 쿼리를 생성하세요.
            
            규칙:
            1. 핵심 키워드를 추출하세요
            2. 불필요한 조사나 어미는 제거하세요
            3. 검색에 도움이 되는 관련 용어나 동의어를 포함하세요
            4. 쿼리는 간결하고 명확해야 합니다
            
            질문: {question}
            
            검색 쿼리:"""),
        ])
        
        self.search_router_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 질문의 특성을 분석하여 최적의 검색 방법을 결정하는 전문가입니다.
            
            다음 중 하나를 선택하세요:
            
            'internal': 
            - Elastic Search 와 관련된 질문
            - RAG와 관련된 질문
            
            'google': 웹 검색이 필요한 경우
            - 최신 정보나 뉴스가 필요한 질문
            - 실시간 데이터나 현재 상황에 대한 질문
            - 특정 기업이나 제품의 최신 정보
            - 일반적인 웹 정보가 도움이 될 수 있는 질문
            
            'arxiv': 학술 논문 검색이 필요한 경우
            - 최신 연구 결과나 학술적 내용
            - 특정 알고리즘이나 기술의 연구 동향
            - 논문이나 연구 결과가 필요한 질문
            - 학술적 근거가 필요한 기술적 질문
            
            질문: {question}
            
            어떤 검색 방법이 가장 적합한가요? (internal/google/arxiv):"""),
        ])
        
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 도움이 되는 AI 어시스턴트입니다. 
            주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공하세요.
            
            규칙:
            1. 내부 문서와 외부 검색 결과를 모두 활용하세요
            2. 정보의 출처를 명확히 구분하세요 (내부 자료 vs 웹 검색 vs 학술 논문)
            3. 확실하지 않은 정보는 추측하지 마세요
            4. 답변은 친근하고 이해하기 쉽게 작성하세요
            5. 필요한 경우 링크나 참고자료를 제공하세요
            6. 컨텍스트에 관련 정보가 없으면 그렇게 말하세요
            
            컨텍스트:
            {context}
            
            질문: {question}
            
            답변:"""),
        ])
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(EnhancedRAGState)
        
        workflow.add_node("query_optimization", self.optimize_query)
        workflow.add_node("search_routing", self.route_search_method)
        workflow.add_node("internal_search", self.search_internal_only)
        workflow.add_node("google_search", self.search_google_only)
        workflow.add_node("arxiv_search", self.search_arxiv_only)
        workflow.add_node("generation", self.generate_answer)
        
        workflow.set_entry_point("query_optimization")
        workflow.add_edge("query_optimization", "search_routing")
        
        # 3-way 조건부 라우팅
        workflow.add_conditional_edges(
            "search_routing",
            self.route_to_search_method,
            {
                "internal": "internal_search",
                "google": "google_search",
                "arxiv": "arxiv_search"
            }
        )
        
        workflow.add_edge("internal_search", "generation")
        workflow.add_edge("google_search", "generation")
        workflow.add_edge("arxiv_search", "generation")
        workflow.add_edge("generation", END)
        
        return workflow.compile()
    
    def optimize_query(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """질문을 검색에 최적화된 쿼리로 변환"""
        try:
            prompt = self.retrieval_prompt.format(question=state["question"])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            optimized_query = response.content.strip()
            
            state["optimized_query"] = optimized_query
            return state
        except Exception as e:
            print(f"Query optimization error: {e}")
            state["optimized_query"] = state["question"]
            return state
    
    def route_search_method(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """검색 방법 결정"""
        try:
            prompt = self.search_router_prompt.format(question=state["question"])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            route = response.content.strip().lower()
            
            # 유효한 라우트인지 확인
            valid_routes = ['internal', 'google', 'arxiv']
            if route not in valid_routes:
                route = 'internal'  # 기본값
                
            state["search_route"] = route
            return state
        except Exception as e:
            print(f"Search routing error: {e}")
            state["search_route"] = "internal"
            return state
    
    def route_to_search_method(self, state: EnhancedRAGState) -> str:
        """라우팅 결정에 따른 분기"""
        return state.get("search_route", "internal")
    
    def search_internal_only(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """내부 Elasticsearch에서만 검색"""
        try:
            query = state.get("optimized_query", state["question"])
            retrieved_docs = self.es_client.internal_search(query)
            
            state["retrieved_docs"] = retrieved_docs
            state["google_results"] = []
            state["arxiv_results"] = []
            return state
        except Exception as e:
            print(f"Internal search error: {e}")
            state["retrieved_docs"] = []
            state["google_results"] = []
            state["arxiv_results"] = []
            return state
    
    def search_google_only(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """내부 검색 + Google 검색"""
        try:
            query = state.get("optimized_query", state["question"])
            
            # 내부 검색
            retrieved_docs = self.es_client.internal_search(query)
            
            # Google 검색
            google_results = self.search_manager.google_tool.search(query, num_results=5)
            
            state["retrieved_docs"] = retrieved_docs
            state["google_results"] = google_results
            state["arxiv_results"] = []
            return state
        except Exception as e:
            print(f"Google search error: {e}")
            state["retrieved_docs"] = []
            state["google_results"] = []
            state["arxiv_results"] = []
            return state
    
    def search_arxiv_only(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """내부 검색 + arXiv 검색"""
        try:
            query = state.get("optimized_query", state["question"])
            
            # 내부 검색
            retrieved_docs = self.es_client.internal_search(query)
            
            # arXiv 검색
            arxiv_results = self.search_manager.arxiv_tool.search(query, max_results=5)
            
            state["retrieved_docs"] = retrieved_docs
            state["google_results"] = []
            state["arxiv_results"] = arxiv_results
            return state
        except Exception as e:
            print(f"ArXiv search error: {e}")
            state["retrieved_docs"] = []
            state["google_results"] = []
            state["arxiv_results"] = []
            return state
    
    def generate_answer(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """통합된 컨텍스트를 바탕으로 답변 생성"""
        try:
            # 내부 문서 컨텍스트 구성
            context_parts = []
            
            if state["retrieved_docs"]:
                context_parts.append("=== 내부 자료 ===")
                for i, doc in enumerate(state["retrieved_docs"], 1):
                    context_parts.append(
                        f"[문서 {i}] (파일: {doc['file_name']}, 점수: {doc['score']:.3f})\n"
                        f"{doc['chunk_text']}\n"
                    )
            
            # Google 검색 결과 추가
            if state.get("google_results"):
                context_parts.append("=== 웹 검색 결과 ===")
                for i, result in enumerate(state["google_results"], 1):
                    context_parts.append(
                        f"[웹 {i}] {result['title']}\n"
                        f"URL: {result['url']}\n"
                        f"내용: {result['snippet']}\n"
                    )
            
            # arXiv 검색 결과 추가
            if state.get("arxiv_results"):
                context_parts.append("=== 학술 논문 검색 결과 ===")
                for i, paper in enumerate(state["arxiv_results"], 1):
                    authors_str = ", ".join(paper['authors'][:3])
                    if len(paper['authors']) > 3:
                        authors_str += " 외"
                    
                    context_parts.append(
                        f"[논문 {i}] {paper['title']}\n"
                        f"저자: {authors_str}\n"
                        f"발행일: {paper['published_date']}\n"
                        f"요약: {paper['summary']}\n"
                        f"PDF: {paper['pdf_url']}\n"
                    )
            
            # 전체 컨텍스트 구성
            if context_parts:
                context = "\n\n".join(context_parts)
            else:
                context = "관련 정보를 찾을 수 없습니다."
            
            state["context"] = context
            
            # 답변 생성
            prompt = self.generation_prompt.format(
                context=context,
                question=state["question"]
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["answer"] = response.content
            
            # 메시지 히스토리 업데이트
            if "messages" not in state:
                state["messages"] = []
                
            state["messages"].extend([
                HumanMessage(content=state["question"]),
                AIMessage(content=state["answer"])
            ])
            
            return state
            
        except Exception as e:
            print(f"Generation error: {e}")
            state["answer"] = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
            return state
    
    def ask(self, question: str) -> Dict[str, Any]:
        """향상된 RAG 파이프라인 실행"""
        initial_state = EnhancedRAGState(
            question=question,
            optimized_query="",
            search_route="",
            retrieved_docs=[],
            google_results=[],
            arxiv_results=[],
            context="",
            answer="",
            messages=[]
        )
        
        result = self.graph.invoke(initial_state)
        
        return {
            "question": result["question"],
            "optimized_query": result.get("optimized_query", ""),
            "search_route": result.get("search_route", ""),
            "answer": result["answer"],
            "context": result["context"],
            "retrieved_docs": result["retrieved_docs"],
            "google_results": result["google_results"],
            "arxiv_results": result["arxiv_results"],
            "internal_source_count": len(result["retrieved_docs"]),
            "google_source_count": len(result["google_results"]),
            "arxiv_source_count": len(result["arxiv_results"])
        }