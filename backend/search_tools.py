import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import time
from urllib.parse import quote_plus
import json

class GoogleSearchTool:
    """구글 검색 API를 통한 웹 검색 도구"""
    
    def __init__(self, api_key: str = None, search_engine_id: str = None):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 5, lang: str = "ko") -> List[Dict[str, Any]]:
        """
        구글 검색 실행
        
        Args:
            query: 검색 쿼리
            num_results: 반환할 결과 수 (최대 10)
            lang: 검색 언어 ('ko', 'en' 등)
        
        Returns:
            검색 결과 리스트
        """
        if not self.api_key or not self.search_engine_id:
            return self._fallback_search(query, num_results)
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),
                'lr': f'lang_{lang}' if lang else None,
                'safe': 'active'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google_search'
                })
            
            return results
            
        except Exception as e:
            print(f"Google Search API error: {e}")
            return self._fallback_search(query, num_results)
    
    def _fallback_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """API 키가 없을 때 사용하는 대체 검색"""
        return [{
            'title': f'Google Search: {query}',
            'url': f'https://www.google.com/search?q={quote_plus(query)}',
            'snippet': f'Google 검색 결과를 확인하려면 위 링크를 클릭하세요. 검색어: {query}',
            'source': 'google_fallback'
        }]


class ArxivSearchTool:
    """arXiv API를 통한 논문 검색 도구"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 5, sort_by: str = "relevance") -> List[Dict[str, Any]]:
        """
        arXiv에서 논문 검색
        
        Args:
            query: 검색 쿼리
            max_results: 반환할 결과 수
            sort_by: 정렬 방식 ('relevance', 'lastUpdatedDate', 'submittedDate')
        
        Returns:
            논문 검색 결과 리스트
        """
        try:
            # arXiv API 파라미터 구성
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': sort_by,
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            # XML 응답 파싱
            root = ET.fromstring(response.content)
            
            # 네임스페이스 정의
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            results = []
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                # 기본 정보 추출
                title = entry.find('atom:title', namespaces)
                title_text = title.text.strip().replace('\n', ' ') if title is not None else ''
                
                summary = entry.find('atom:summary', namespaces)
                summary_text = summary.text.strip().replace('\n', ' ')[:300] + '...' if summary is not None else ''
                
                # 저자 정보
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name = author.find('atom:name', namespaces)
                    if name is not None:
                        authors.append(name.text)
                
                # URL 정보
                pdf_url = ""
                abs_url = ""
                for link in entry.findall('atom:link', namespaces):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href', '')
                    elif link.get('rel') == 'alternate':
                        abs_url = link.get('href', '')
                
                # 발행 날짜
                published = entry.find('atom:published', namespaces)
                published_date = published.text[:10] if published is not None else ''
                
                # 카테고리
                categories = []
                for category in entry.findall('atom:category', namespaces):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                
                result = {
                    'title': title_text,
                    'authors': authors,
                    'summary': summary_text,
                    'pdf_url': pdf_url,
                    'abs_url': abs_url,
                    'published_date': published_date,
                    'categories': categories,
                    'source': 'arxiv'
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []
    
    def search_by_category(self, category: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        특정 카테고리의 최신 논문 검색
        
        Args:
            category: arXiv 카테고리 (예: 'cs.AI', 'cs.CL', 'stat.ML')
            max_results: 반환할 결과 수
        
        Returns:
            논문 검색 결과 리스트
        """
        try:
            params = {
                'search_query': f'cat:{category}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            # 기존 search 메서드와 동일한 파싱 로직 사용
            root = ET.fromstring(response.content)
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            results = []
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                title = entry.find('atom:title', namespaces)
                title_text = title.text.strip().replace('\n', ' ') if title is not None else ''
                
                summary = entry.find('atom:summary', namespaces)
                summary_text = summary.text.strip().replace('\n', ' ')[:300] + '...' if summary is not None else ''
                
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name = author.find('atom:name', namespaces)
                    if name is not None:
                        authors.append(name.text)
                
                pdf_url = ""
                abs_url = ""
                for link in entry.findall('atom:link', namespaces):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href', '')
                    elif link.get('rel') == 'alternate':
                        abs_url = link.get('href', '')
                
                published = entry.find('atom:published', namespaces)
                published_date = published.text[:10] if published is not None else ''
                
                categories = []
                for cat in entry.findall('atom:category', namespaces):
                    term = cat.get('term')
                    if term:
                        categories.append(term)
                
                result = {
                    'title': title_text,
                    'authors': authors,
                    'summary': summary_text,
                    'pdf_url': pdf_url,
                    'abs_url': abs_url,
                    'published_date': published_date,
                    'categories': categories,
                    'source': 'arxiv_category'
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"ArXiv category search error: {e}")
            return []


class SearchToolsManager:
    """검색 도구들을 관리하는 매니저 클래스"""
    
    def __init__(self, google_api_key: str = None, google_search_engine_id: str = None):
        self.google_tool = GoogleSearchTool(google_api_key, google_search_engine_id)
        self.arxiv_tool = ArxivSearchTool()
    
    def search_all(self, query: str, google_results: int = 3, arxiv_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        구글과 arXiv 동시 검색
        
        Args:
            query: 검색 쿼리
            google_results: 구글 검색 결과 수
            arxiv_results: arXiv 검색 결과 수
        
        Returns:
            검색 결과를 소스별로 분류한 딕셔너리
        """
        results = {
            'google': [],
            'arxiv': [],
            'total_count': 0
        }
        
        try:
            # 구글 검색
            google_results_data = self.google_tool.search(query, google_results)
            results['google'] = google_results_data
            
            # arXiv 검색
            arxiv_results_data = self.arxiv_tool.search(query, arxiv_results)
            results['arxiv'] = arxiv_results_data
            
            results['total_count'] = len(google_results_data) + len(arxiv_results_data)
            
        except Exception as e:
            print(f"Search all error: {e}")
        
        return results
    
    def format_results_for_context(self, search_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        검색 결과를 RAG 컨텍스트용으로 포맷팅
        
        Args:
            search_results: 검색 결과 딕셔너리
        
        Returns:
            포맷팅된 컨텍스트 문자열
        """
        context_parts = []
        
        # 구글 검색 결과
        if search_results.get('google'):
            context_parts.append("=== 웹 검색 결과 ===")
            for i, result in enumerate(search_results['google'], 1):
                context_parts.append(
                    f"[웹 {i}] {result['title']}\n"
                    f"URL: {result['url']}\n"
                    f"내용: {result['snippet']}\n"
                )
        
        # arXiv 검색 결과
        if search_results.get('arxiv'):
            context_parts.append("=== 학술 논문 검색 결과 ===")
            for i, result in enumerate(search_results['arxiv'], 1):
                authors_str = ", ".join(result['authors'][:3])  # 처음 3명만
                if len(result['authors']) > 3:
                    authors_str += " 외"
                
                context_parts.append(
                    f"[논문 {i}] {result['title']}\n"
                    f"저자: {authors_str}\n"
                    f"발행일: {result['published_date']}\n"
                    f"요약: {result['summary']}\n"
                    f"PDF: {result['pdf_url']}\n"
                )
        
        return "\n".join(context_parts) if context_parts else "관련 검색 결과를 찾을 수 없습니다."