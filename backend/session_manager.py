#!/usr/bin/env python3
"""
대화 세션 관리 시스템
멀티턴 대화를 위한 세션 상태 저장 및 관리
"""

import uuid
import time
from typing import Dict, List, Any, Optional
from threading import Lock
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

@dataclass
class ConversationSession:
    """대화 세션 데이터 클래스"""
    session_id: str
    created_at: float
    last_accessed: float
    messages: List[BaseMessage] = field(default_factory=list)
    context_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionManager:
    """대화 세션 관리자"""
    
    def __init__(self, session_timeout: int = 3600):  # 1시간 기본 타임아웃
        self.sessions: Dict[str, ConversationSession] = {}
        self.session_timeout = session_timeout
        self.lock = Lock()
    
    def create_session(self) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        current_time = time.time()
        
        with self.lock:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                created_at=current_time,
                last_accessed=current_time
            )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """세션 조회"""
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # 세션 만료 확인
            if time.time() - session.last_accessed > self.session_timeout:
                del self.sessions[session_id]
                return None
            
            # 접근 시간 업데이트
            session.last_accessed = time.time()
            return session
    
    def add_message(self, session_id: str, message: BaseMessage) -> bool:
        """세션에 메시지 추가"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        with self.lock:
            session.messages.append(message)
            # 최대 20개 메시지만 유지 (메모리 관리)
            if len(session.messages) > 20:
                session.messages = session.messages[-20:]
        
        return True
    
    def add_conversation_turn(
        self, 
        session_id: str, 
        question: str, 
        answer: str, 
        context: str = ""
    ) -> bool:
        """대화 턴 추가 (질문-답변 쌍)"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        with self.lock:
            # 메시지 히스토리에 추가
            session.messages.extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            
            # 컨텍스트 히스토리에 추가
            if context:
                session.context_history.append(context)
                # 최대 5개 컨텍스트만 유지
                if len(session.context_history) > 5:
                    session.context_history = session.context_history[-5:]
            
            # 메시지 개수 제한
            if len(session.messages) > 20:
                session.messages = session.messages[-20:]
        
        return True
    
    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """대화 히스토리 조회"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        return list(session.messages)
    
    def get_context_summary(self, session_id: str) -> str:
        """이전 대화의 컨텍스트 요약"""
        session = self.get_session(session_id)
        if not session or not session.messages:
            return ""
        
        # 최근 3턴의 대화만 사용
        recent_messages = session.messages[-6:]  # 3턴 = 6개 메시지 (질문+답변)
        
        if not recent_messages:
            return ""
        
        context_parts = []
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                question = recent_messages[i].content
                answer = recent_messages[i + 1].content
                context_parts.append(f"이전 질문: {question}\n이전 답변: {answer[:200]}...")
        
        return "\n\n".join(context_parts)
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        current_time = time.time()
        expired_sessions = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                if current_time - session.last_accessed > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
        
        return len(expired_sessions)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계 정보"""
        with self.lock:
            active_sessions = len(self.sessions)
            total_messages = sum(len(session.messages) for session in self.sessions.values())
            
            return {
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "session_timeout": self.session_timeout
            }

# 전역 세션 매니저 인스턴스
session_manager = SessionManager()