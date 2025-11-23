from typing import List, Dict, Any, TypedDict
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from EsClient import EsClient

load_dotenv()

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    answer: str
    messages: List[Any]

class RAGGraph:
    def __init__(self):
        self.es_client = EsClient()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_PROXY_URL")
        )
        
        self.retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 질문을 분석하여 검색에 최적화된 쿼리를 생성하는 전문가입니다.
            사용자의 질문을 받아서 Elasticsearch에서 검색할 수 있는 효과적인 쿼리를 생성하세요.
            
            규칙:
            1. 핵심 키워드를 추출하세요
            2. 불필요한 조사나 어미는 제거하세요
            3. 검색에 도움이 되는 관련 용어나 동의어를 포함하세요
            4. 쿼리는 간결하고 명확해야 합니다
            
            질문: {question}
            
            검색 쿼리:"""),
        ])
        
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 도움이 되는 AI 어시스턴트입니다. 
            주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공하세요.
            
            규칙:
            1. 컨텍스트에 있는 정보만 사용하여 답변하세요
            2. 확실하지 않은 정보는 추측하지 마세요
            3. 답변은 친근하고 이해하기 쉽게 작성하세요
            4. 필요한 경우 예시나 설명을 포함하세요
            5. 컨텍스트에 관련 정보가 없으면 그렇게 말하세요
            
            컨텍스트:
            {context}
            
            질문: {question}
            
            답변:"""),
        ])
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RAGState)
        
        workflow.add_node("query_optimization", self.optimize_query)
        workflow.add_node("retrieval", self.retrieve_documents)
        workflow.add_node("generation", self.generate_answer)
        
        workflow.set_entry_point("query_optimization")
        workflow.add_edge("query_optimization", "retrieval")
        workflow.add_edge("retrieval", "generation")
        workflow.add_edge("generation", END)
        
        return workflow.compile()
    
    def optimize_query(self, state: RAGState) -> RAGState:
        """질문을 검색에 최적화된 쿼리로 변환"""
        try:
            prompt = self.retrieval_prompt.format(question=state["question"])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            optimized_query = response.content.strip()
            
            # 최적화된 쿼리로 질문을 업데이트 (검색용)
            state["optimized_query"] = optimized_query
            return state
        except Exception as e:
            print(f"Query optimization error: {e}")
            state["optimized_query"] = state["question"]  # 원본 질문 사용
            return state
    
    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Elasticsearch에서 관련 문서 검색"""
        try:
            query = state.get("optimized_query", state["question"])
            retrieved_docs = self.es_client.internal_search(query)
            
            state["retrieved_docs"] = retrieved_docs
            return state
        except Exception as e:
            print(f"Retrieval error: {e}")
            state["retrieved_docs"] = []
            return state
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """검색된 문서를 바탕으로 답변 생성"""
        try:
            # 컨텍스트 구성
            context_parts = []
            for i, doc in enumerate(state["retrieved_docs"], 1):
                context_parts.append(
                    f"[문서 {i}] (파일: {doc['file_name']}, 점수: {doc['score']:.3f})\n"
                    f"{doc['chunk_text']}\n"
                )
            
            context = "\n".join(context_parts) if context_parts else "관련 정보를 찾을 수 없습니다."
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
        """RAG 파이프라인 실행"""
        initial_state = RAGState(
            question=question,
            retrieved_docs=[],
            context="",
            answer="",
            messages=[]
        )
        
        result = self.graph.invoke(initial_state)
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": result["context"],
            "retrieved_docs": result["retrieved_docs"],
            "source_count": len(result["retrieved_docs"])
        }