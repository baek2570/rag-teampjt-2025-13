from typing import List, Dict, Any, TypedDict, Optional
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from es_client import EsClient
from search_tools import SearchToolsManager
from improved_prompts import (
    INTENT_DETECTION_SYSTEM,
    QUERY_OPTIMIZATION_SYSTEM,
    RELEVANCE_EVALUATION_SYSTEM,
    EXTERNAL_SEARCH_ROUTER_SYSTEM,
    ANSWER_GENERATION_SYSTEM
)

load_dotenv()

class EnhancedRAGState(TypedDict):
    question: str
    user_intent: str
    optimized_query: str
    search_route: str
    retrieved_docs: List[Dict[str, Any]]
    google_results: List[Dict[str, Any]]
    arxiv_results: List[Dict[str, Any]]
    internal_relevance_score: float
    needs_external_search: bool
    external_search_type: str
    context: str
    answer: str
    messages: List[Any]
    conversation_history: List[BaseMessage]
    session_context: str

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

        self.intent_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_DETECTION_SYSTEM),
        ])

        self.retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_OPTIMIZATION_SYSTEM),
        ])

        self.relevance_evaluator_prompt = ChatPromptTemplate.from_messages([
            ("system", RELEVANCE_EVALUATION_SYSTEM),
        ])

        self.external_search_router_prompt = ChatPromptTemplate.from_messages([
            ("system", EXTERNAL_SEARCH_ROUTER_SYSTEM),
        ])

        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_GENERATION_SYSTEM),
        ])
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(EnhancedRAGState)

        workflow.add_node("intent_detection", self.detect_intent)
        workflow.add_node("query_optimization", self.optimize_query)
        workflow.add_node("internal_search", self.search_internal_only)
        workflow.add_node("relevance_evaluation", self.evaluate_relevance)
        workflow.add_node("external_search_routing", self.route_external_search)
        workflow.add_node("google_search", self.search_google_additional)
        workflow.add_node("arxiv_search", self.search_arxiv_additional)
        workflow.add_node("generation", self.generate_answer)

        workflow.set_entry_point("intent_detection")
        workflow.add_edge("intent_detection", "query_optimization")
        workflow.add_edge("query_optimization", "internal_search")
        workflow.add_edge("internal_search", "relevance_evaluation")
        
        # 관련성 평가 후 조건부 라우팅
        workflow.add_conditional_edges(
            "relevance_evaluation",
            self.decide_next_step,
            {
                "sufficient": "generation",
                "need_external": "external_search_routing"
            }
        )
        
        # 외부 검색 타입 결정
        workflow.add_conditional_edges(
            "external_search_routing",
            self.route_to_external_search,
            {
                "google": "google_search",
                "arxiv": "arxiv_search"
            }
        )
        
        workflow.add_edge("google_search", "generation")
        workflow.add_edge("arxiv_search", "generation")
        workflow.add_edge("generation", END)
        
        return workflow.compile()

    def detect_intent(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """사용자의 검색 의도 파악"""
        try:
            prompt = self.intent_detection_prompt.format(question=state["question"])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            intent = response.content.strip().lower()

            valid_intents = ['explicit_google', 'explicit_arxiv', 'internal_only', 'flexible']
            if intent not in valid_intents:
                intent = 'flexible'

            state["user_intent"] = intent
            return state
        except Exception as e:
            print(f"Intent detection error: {e}")
            state["user_intent"] = "flexible"
            return state

    def optimize_query(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """질문을 검색에 최적화된 쿼리로 변환"""
        try:
            session_context = state.get("session_context", "이전 대화 없음")
            prompt = self.retrieval_prompt.format(
                question=state["question"],
                session_context=session_context
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            optimized_query = response.content.strip()
            
            state["optimized_query"] = optimized_query
            return state
        except Exception as e:
            print(f"Query optimization error: {e}")
            state["optimized_query"] = state["question"]
            return state
    
    def evaluate_relevance(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """내부 검색 결과의 관련성 평가"""
        try:
            # 검색 결과를 텍스트로 포맷팅
            if state["retrieved_docs"]:
                search_results_text = "\n\n".join([
                    f"[문서 {i+1}] \n{doc['chunk_text']}"
                    for i, doc in enumerate(state["retrieved_docs"])
                ])
            else:
                search_results_text = "검색 결과가 없습니다."
            
            prompt = self.relevance_evaluator_prompt.format(
                question=state["question"],
                search_results=search_results_text
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                score = float(response.content.strip())
                score = max(0, min(10, score))  # 0~10 범위로 제한
            except ValueError:
                score = 3.0  # 파싱 실패 시 기본값
            
            state["internal_relevance_score"] = score
            state["needs_external_search"] = score <= 5.0
            
            if score > 5.0:
                state["search_route"] = "internal_sufficient"
            else:
                state["search_route"] = "need_external"
                
            return state
        except Exception as e:
            print(f"Relevance evaluation error: {e}")
            state["internal_relevance_score"] = 3.0
            state["needs_external_search"] = True
            state["search_route"] = "need_external"
            return state
    
    def decide_next_step(self, state: EnhancedRAGState) -> str:
        """관련성 평가 결과에 따른 다음 단계 결정"""
        if state.get("needs_external_search", True):
            return "need_external"
        else:
            return "sufficient"
    
    def route_external_search(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """외부 검색 타입 결정 (user_intent 우선)"""
        try:
            user_intent = state.get("user_intent", "flexible")

            # 사용자가 명시적으로 검색 도구를 요청한 경우
            if user_intent == "explicit_google":
                state["external_search_type"] = "google"
            elif user_intent == "explicit_arxiv":
                state["external_search_type"] = "arxiv"
            elif user_intent == "internal_only":
                # 내부 검색만 원하는 경우에도 외부 검색이 필요하다면 google을 기본으로
                state["external_search_type"] = "google"
            else:
                # flexible인 경우 LLM이 판단
                prompt = self.external_search_router_prompt.format(
                    question=state["question"],
                    user_intent=user_intent
                )
                response = self.llm.invoke([HumanMessage(content=prompt)])
                external_type = response.content.strip().lower()

                if external_type not in ['google', 'arxiv']:
                    external_type = 'google'

                state["external_search_type"] = external_type

            return state
        except Exception as e:
            print(f"External search routing error: {e}")
            state["external_search_type"] = "google"
            return state
    
    def route_to_external_search(self, state: EnhancedRAGState) -> str:
        """외부 검색 타입에 따른 분기"""
        return state.get("external_search_type", "google")
    
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
    
    def search_google_additional(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Google 검색 추가 수행 (내부 검색은 이미 완료됨)"""
        try:
            query = state.get("optimized_query", state["question"])
            
            # Google 검색
            google_results = self.search_manager.google_tool.search(query, num_results=5)
            
            state["google_results"] = google_results
            state["arxiv_results"] = []
            state["retrieved_docs"] = []
            return state
        except Exception as e:
            print(f"Google search error: {e}")
            state["google_results"] = []
            state["arxiv_results"] = []
            return state
    
    def search_arxiv_additional(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """arXiv 검색 추가 수행 (내부 검색은 이미 완료됨)"""
        try:
            query = state.get("optimized_query", state["question"])
            
            # arXiv 검색
            arxiv_results = self.search_manager.arxiv_tool.search(query, max_results=5)
            
            state["google_results"] = []
            state["arxiv_results"] = arxiv_results
            state["retrieved_docs"] = []
            return state
        except Exception as e:
            print(f"ArXiv search error: {e}")
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
            session_context = state.get("session_context", "이전 대화 없음")
            prompt = self.generation_prompt.format(
                context=context,
                question=state["question"],
                session_context=session_context
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
    
    def ask(self, question: str, conversation_history: Optional[List[BaseMessage]] = None, session_context: str = "") -> Dict[str, Any]:
        """향상된 RAG 파이프라인 실행 (멀티턴 대화 지원)"""
        initial_state = EnhancedRAGState(
            question=question,
            user_intent="",
            optimized_query="",
            search_route="",
            retrieved_docs=[],
            google_results=[],
            arxiv_results=[],
            internal_relevance_score=0.0,
            needs_external_search=False,
            external_search_type="",
            context="",
            answer="",
            messages=[],
            conversation_history=conversation_history or [],
            session_context=session_context
        )
        
        result = self.graph.invoke(initial_state)

        return {
            "question": result["question"],
            "user_intent": result.get("user_intent", ""),
            "optimized_query": result.get("optimized_query", ""),
            "search_route": result.get("search_route", ""),
            "answer": result["answer"],
            "context": result["context"],
            "retrieved_docs": result["retrieved_docs"],
            "google_results": result["google_results"],
            "arxiv_results": result["arxiv_results"],
            "internal_source_count": len(result["retrieved_docs"]),
            "google_source_count": len(result["google_results"]),
            "arxiv_source_count": len(result["arxiv_results"]),
            "conversation_history": result.get("conversation_history", []),
            "session_context": result.get("session_context", "")
        }