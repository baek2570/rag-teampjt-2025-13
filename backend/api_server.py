#!/usr/bin/env python3
"""
FastAPI 기반 RAG 시스템 API 서버
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_graph import EnhancedRAGGraph
from session_manager import session_manager

# 환경 변수 로드
load_dotenv()

# 전역 RAG 시스템 인스턴스
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 라이프사이클 관리"""
    global rag_system
    
    # 시작 시 RAG 시스템 초기화
    try:
        print("🚀 RAG 시스템 초기화 중...")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        rag_system = EnhancedRAGGraph(google_api_key, google_search_engine_id)
        print("✅ RAG 시스템 초기화 완료")
    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {e}")
        traceback.print_exc()
        rag_system = None
    
    yield
    
    # 종료 시 정리 작업 (필요한 경우)
    print("🛑 서버 종료 중...")

# FastAPI 앱 생성
app = FastAPI(
    title="Enhanced RAG API",
    description="내부 검색 우선, 외부 검색 폴백을 지원하는 RAG 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    
class AnswerResponse(BaseModel):
    question: str
    answer: str
    optimized_query: str
    search_route: str
    internal_source_count: int
    google_source_count: int
    arxiv_source_count: int
    context: str
    session_id: str
    
class HealthResponse(BaseModel):
    status: str
    message: str
    rag_system_ready: bool

class SessionCreateResponse(BaseModel):
    session_id: str
    message: str

class SessionStatsResponse(BaseModel):
    session_stats: Dict[str, Any]
    message: str

# API 엔드포인트
@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Enhanced RAG API 서버",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthResponse(
        status="healthy" if rag_system is not None else "unhealthy",
        message="RAG 시스템이 준비되었습니다." if rag_system is not None else "RAG 시스템 초기화에 실패했습니다.",
        rag_system_ready=rag_system is not None
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """질문 처리 엔드포인트 (멀티턴 대화 지원)"""
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 준비되지 않았습니다. 서버를 다시 시작해 주세요."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문이 비어있습니다."
        )
    
    try:
        # 세션 ID 처리
        session_id = request.session_id
        if not session_id:
            # 새 세션 생성
            session_id = session_manager.create_session()
            print(f"🆕 새 세션 생성: {session_id}")
        
        print(f"📝 질문 수신 (세션: {session_id[:8]}...): {request.question}")
        
        # 대화 히스토리 조회
        conversation_history = session_manager.get_conversation_history(session_id)
        session_context = session_manager.get_context_summary(session_id)
        
        # RAG 시스템으로 질문 처리 (대화 맥락 포함)
        result = rag_system.ask(
            request.question,
            conversation_history=conversation_history,
            session_context=session_context
        )
        
        # 세션에 대화 턴 저장
        session_manager.add_conversation_turn(
            session_id,
            request.question,
            result["answer"],
            result.get("context", "")
        )
        
        response = AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            optimized_query=result.get("optimized_query", ""),
            search_route=result.get("search_route", ""),
            internal_source_count=result.get("internal_source_count", 0),
            google_source_count=result.get("google_source_count", 0),
            arxiv_source_count=result.get("arxiv_source_count", 0),
            context=result.get("context", ""),
            session_id=session_id
        )
        
        print(f"✅ 답변 완료 (세션: {session_id[:8]}...) - 내부:{response.internal_source_count}, 구글:{response.google_source_count}, arXiv:{response.arxiv_source_count}")
        
        return response
        
    except Exception as e:
        print(f"❌ 질문 처리 중 오류: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"질문 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """시스템 통계 정보"""
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 준비되지 않았습니다."
        )
    
    return {
        "system_status": "ready",
        "available_search_methods": ["internal", "google", "arxiv"],
        "description": {
            "internal": "내부 Elasticsearch 검색",
            "google": "Google 웹 검색", 
            "arxiv": "arXiv 학술 논문 검색"
        },
        "flow": [
            "1. 내부 검색 우선 실행",
            "2. 결과 있으면 바로 답변 생성",
            "3. 결과 없으면 외부 검색으로 폴백",
            "4. 질문 특성에 따라 Google 또는 arXiv 선택"
        ]
    }

@app.post("/session/create", response_model=SessionCreateResponse)
async def create_session():
    """새 대화 세션 생성"""
    try:
        session_id = session_manager.create_session()
        return SessionCreateResponse(
            session_id=session_id,
            message="새 대화 세션이 생성되었습니다."
        )
    except Exception as e:
        print(f"❌ 세션 생성 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"세션 생성 중 오류가 발생했습니다: {str(e)}"
        )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """대화 세션 삭제"""
    try:
        success = session_manager.delete_session(session_id)
        if success:
            return {"message": "세션이 삭제되었습니다.", "session_id": session_id}
        else:
            raise HTTPException(
                status_code=404,
                detail="세션을 찾을 수 없습니다."
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 세션 삭제 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"세션 삭제 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/session/stats", response_model=SessionStatsResponse)
async def get_session_stats():
    """세션 통계 정보"""
    try:
        # 만료된 세션 정리
        cleaned_count = session_manager.cleanup_expired_sessions()
        if cleaned_count > 0:
            print(f"🧹 만료된 세션 {cleaned_count}개 정리됨")
        
        stats = session_manager.get_session_stats()
        return SessionStatsResponse(
            session_stats=stats,
            message="세션 통계 정보입니다."
        )
    except Exception as e:
        print(f"❌ 세션 통계 조회 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"세션 통계 조회 중 오류가 발생했습니다: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # 환경 변수에서 포트 설정 (기본값: 8000)
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🌟 Enhanced RAG API 서버 시작")
    print(f"📡 주소: http://{host}:{port}")
    print(f"📖 API 문서: http://{host}:{port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )