from typing import List, Dict, Any, TypedDict
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from EsClient import EsClient
from search_tools import SearchToolsManager

load_dotenv()

class EnhancedRAGState(TypedDict):
    question: str
    optimized_query: str
    retrieved_docs: List[Dict[str, Any]]
    external_search_results: Dict[str, List[Dict[str, Any]]]
    context: str
    answer: str
    messages: List[Any]
    use_external_search: bool

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
        
        self.search_decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 질문의 특성을 분석하여 외부 검색이 필요한지 판단하는 전문가입니다.
            
            다음 경우에는 외부 검색이 필요합니다:
            1. 최신 정보나 뉴스가 필요한 질문
            2. 실시간 데이터나 현재 상황에 대한 질문
            3. 특정 기술의 최신 동향이나 발전사항
            4. 학술 논문이나 연구 결과가 필요한 질문
            5. 일반적인 웹 정보가 도움이 될 수 있는 질문
            
            다음 경우에는 내부 데이터만으로 충분합니다:
            1. 검색 엔진과 관련된 질문
            2. Rag와 관련된 질문
            3. 일반적인 설명이나 이론적 내용
            
            질문: {question}
            
            외부 검색이 필요한가요? (yes/no):"""),
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
        workflow.add_node("search_decision", self.decide_external_search)
        workflow.add_node("internal_retrieval", self.retrieve_internal_documents)
        workflow.add_node("external_search", self.search_external_sources)
        workflow.add_node("generation", self.generate_answer)
        
        workflow.set_entry_point("query_optimization")
        workflow.add_edge("query_optimization", "search_decision")
        
        # 조건부 라우팅: 외부 검색 필요 여부에 따라 분기
        workflow.add_conditional_edges(
            "search_decision",
            self.route_after_decision,
            {
                "external_search": "external_search",
                "internal_only": "internal_retrieval"
            }
        )
        
        workflow.add_edge("external_search", "generation")
        workflow.add_edge("internal_retrieval", "generation")
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
    
    def decide_external_search(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """외부 검색 필요 여부 결정"""
        try:
            prompt = self.search_decision_prompt.format(question=state["question"])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            decision = response.content.strip().lower()
            
            state["use_external_search"] = decision == "yes"
            return state
        except Exception as e:
            print(f"Search decision error: {e}")
            state["use_external_search"] = False
            return state
    
    def route_after_decision(self, state: EnhancedRAGState) -> str:
        """검색 결정에 따른 라우팅"""
        return "external_search" if state.get("use_external_search", False) else "internal_only"
    
    def retrieve_internal_documents(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """내부 Elasticsearch에서만 문서 검색"""
        try:
            query = state.get("optimized_query", state["question"])
            retrieved_docs = self.es_client.internal_search(query)
            
            state["retrieved_docs"] = retrieved_docs
            state["external_search_results"] = {"google": [], "arxiv": []}
            return state
        except Exception as e:
            print(f"Internal retrieval error: {e}")
            state["retrieved_docs"] = []
            state["external_search_results"] = {"google": [], "arxiv": []}
            return state
    
    def search_external_sources(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """내부 검색 + 외부 검색 수행"""
        try:
            query = state.get("optimized_query", state["question"])
            
            # 내부 검색
            retrieved_docs = self.es_client.internal_search(query)
            
            # 외부 검색 (구글 + arXiv)
            external_results = self.search_manager.search_all(
                query, 
                google_results=3, 
                arxiv_results=3
            )
            
            state["retrieved_docs"] = retrieved_docs
            state["external_search_results"] = external_results
            return state
        except Exception as e:
            print(f"External search error: {e}")
            state["retrieved_docs"] = []
            state["external_search_results"] = {"google": [], "arxiv": []}
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
            
            # 외부 검색 결과 컨텍스트 추가
            if state.get("external_search_results"):
                external_context = self.search_manager.format_results_for_context(
                    state["external_search_results"]
                )
                if external_context and external_context != "관련 검색 결과를 찾을 수 없습니다.":
                    context_parts.append(external_context)
            
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
            retrieved_docs=[],
            external_search_results={"google": [], "arxiv": []},
            context="",
            answer="",
            messages=[],
            use_external_search=False
        )
        
        result = self.graph.invoke(initial_state)
        
        return {
            "question": result["question"],
            "optimized_query": result.get("optimized_query", ""),
            "answer": result["answer"],
            "context": result["context"],
            "retrieved_docs": result["retrieved_docs"],
            "external_search_results": result["external_search_results"],
            "used_external_search": result.get("use_external_search", False),
            "internal_source_count": len(result["retrieved_docs"]),
            "external_source_count": len(result["external_search_results"].get("google", [])) + len(result["external_search_results"].get("arxiv", []))
        }