import json
import logging
import re
from typing import Dict, List, Optional, Any

import arxiv
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..prompts import PROMPTS
from ..utils import safe_requests_get, make_text_preview, summarize_source_relationships, invoke_json_llm, parse_llm_json_response

logger = logging.getLogger(__name__)

class PaperAgent:
    """
    Unified paper processing agent
    
    Merged functionality from PaperAgent and PDFPaperAgent,
    focused on extracting dataset information from papers
    """
    
    def __init__(self, llm: ChatOpenAI, pdf_crop: bool = False):
        self.llm = llm
        self.pdf_crop = pdf_crop  # When True, crop PDF body by TOC; otherwise use the full PDF.
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def process_paper(self, dataset_name: str, paper_url: str) -> Dict[str, Any]:
        """
        Process single paper and extract dataset information
        
        Args:
            dataset_name: Target dataset name
            paper_url: Paper URL (supports arXiv, PDF links, etc.)
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            paper_url = self._convert_hf_papers_url_to_arxiv(paper_url)
            logger.info(f"Processing paper: {paper_url}")
            
            paper_content = self._fetch_paper_content(paper_url)
            if paper_content == "" or paper_content is None:
                return {"datasets": [], "error": "Failed to fetch paper content"}
            
            dataset_analysis = self._analyze_paper_for_datasets(dataset_name, paper_content)
            if not isinstance(dataset_analysis, list):
                dataset_analysis = []
            return {"datasets": dataset_analysis, "error": None}
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_url}: {str(e)}")
            return {"datasets": [], "error": str(e)}
    
    def _fetch_paper_content(self, paper_url: str) -> str:
        """Unified paper content fetching method"""
        try:
            if paper_url.endswith('.pdf') or 'pdf' in paper_url.lower() or "arxiv.org" in paper_url:
                if "arxiv.org" in paper_url and "abs" in paper_url:
                    paper_url = paper_url.replace("abs", "pdf")
                return self._extract_pdf_content(paper_url)
            else:
                return self._fetch_html_content(paper_url)
                
        except Exception as e:
            logger.error(f"Error fetching paper content: {str(e)}")
            return ""
    
    def _filter_with_toc_from_doc(self, doc) -> str:
        toc = doc.get_toc(simple=True)
        n_pages = len(doc)

        def norm(s):
            return re.sub(r'\s+', ' ', s.strip().lower())

        intro_pages = []
        related_pages = []
        background_pages = []
        exp_pages = []
        eval_pages = []
        concl_pages = []
        future_pages = []

        for _, title, page in toc:
            t = norm(title)
            if re.search(r'\bintroduction\b', t):
                intro_pages.append(page - 1)
            if re.search(r'\brelated\b', t):
                related_pages.append(page - 1)
            if re.search(r'\bbackground\b', t):
                background_pages.append(page - 1)
            if re.search(r'\bexperiment(al)?\b', t):
                exp_pages.append(page - 1)
            if re.search(r'\bevaluation\b', t):
                eval_pages.append(page - 1)
            if re.search(r'\bconclusion\b', t):
                concl_pages.append(page - 1)
            if re.search(r'\bfuture?\b', t):
                future_pages.append(page - 1)

        if concl_pages or future_pages:
            max_page = 0
            if concl_pages:
                max_page = max(max_page, max(concl_pages))
            if future_pages:
                max_page = max(max_page, max(future_pages))
            half_n = max_page // 2
        else:
            half_n = n_pages // 2

        candidates = []
        if related_pages:
            related_in_half = [p for p in related_pages if p < half_n]
            if related_in_half:
                candidates.append(max(related_in_half))
        if intro_pages:
            intro_in_half = [p for p in intro_pages if p < half_n]
            if intro_in_half:
                candidates.append(max(intro_in_half))
        if background_pages:
            background_in_half = [p for p in background_pages if p < half_n]
            if background_in_half:
                candidates.append(max(background_in_half))
        if candidates:
            rm_front_until = max(candidates)
        else:
            rm_front_until = 0

        tail_candidates = []
        if exp_pages:
            tail_candidates.append(min(exp_pages))
        if eval_pages:
            tail_candidates.append(min(eval_pages))
        if concl_pages:
            tail_candidates.append(min(concl_pages))
        if future_pages:
            tail_candidates.append(min(future_pages))

        if tail_candidates:
            rm_after = min(tail_candidates)
        else:
            rm_after = n_pages
        logger.info(f"rm_front_until: {rm_front_until}, rm_after: {rm_after}")
        kept_pages = []
        for i in range(rm_front_until, min(rm_after+1, n_pages)):
            pg = doc[i]
            kept_pages.append(pg.get_text())
        kept_text = "\n".join(kept_pages)

        return kept_text


    def _extract_pdf_content(self, pdf_url: str) -> str:
        """Extract PDF content using optimal method, supports GitHub PDF links"""

        logger.info(f"Extracting PDF content from: {pdf_url}")
        
        try:
            response = safe_requests_get(pdf_url, timeout=30, max_retries=5, sleep_time=30,
                                       headers=self.session.headers, proxies=self.session.proxies)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            is_pdf = content_type and 'application/pdf' in content_type
            if not is_pdf and response.content.startswith(b'%PDF'):
                is_pdf = True
                
            if not is_pdf and 'github.com' in pdf_url and '/blob/' in pdf_url:
                logger.info("Detected GitHub blob URL, trying to download PDF directly")
                download_url = pdf_url.replace('/blob/', '/raw/')
                try:
                    download_response = safe_requests_get(download_url, timeout=30, max_retries=5, sleep_time=30,
                                                       headers=self.session.headers, proxies=self.session.proxies)
                    download_response.raise_for_status()
                    if (download_response.headers.get('content-type', '').startswith('application/pdf') or 
                        download_response.content.startswith(b'%PDF')):
                        response = download_response
                        logger.info("Successfully downloaded PDF from GitHub")
                    else:
                        logger.warning("Downloaded content is not a PDF")
                except Exception as e:
                    logger.warning(f"Failed to download from GitHub: {e}")
            
            try:
                import fitz
                doc = fitz.open(stream=response.content, filetype="pdf")
                full_text = ""
                if self.pdf_crop and len(doc) > 3:
                    filtered_text = self._filter_with_toc_from_doc(doc)
                else:
                    for page in doc:
                        text = page.get_text()
                        if text.strip():
                            full_text += text + "\n"
                    filtered_text = full_text
                doc.close()

                if filtered_text.strip():
                    return filtered_text
                else:
                    return None
                    
            except ImportError:
                logger.warning("PyMuPDF not available, PDF extraction skipped")
            except Exception as e:
                logger.error(f"PyMuPDF extraction failed: {e}")
                
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return ""
            
    def _fetch_html_content(self, url: str) -> str:
        """Fetch HTML page content"""
        try:
            response = safe_requests_get(url, timeout=30, max_retries=5, sleep_time=30,
                                       headers=self.session.headers, proxies=self.session.proxies)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = ""
            title_selectors = ['h1', 'title', '.title', '#title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            content = soup.get_text(separator='\n')
            
            return content
            
        except Exception as e:
            logger.error(f"Error fetching HTML content: {str(e)}")
            return ""
    
    # ================================================================
    # ================================================================
    
    def _analyze_paper_for_datasets(self, dataset_name: str, paper_content: str) -> List[Dict[str, Any]]:
        """Analyze paper content using LLM to extract dataset information"""
        try:
            logger.info(f"Analyzing paper for datasets: {dataset_name}")
            logger.info(
                f"Paper content preview for {dataset_name}: "
                f"{make_text_preview(paper_content, max_len=600)}"
            )
            prompt = ChatPromptTemplate.from_template(PROMPTS['pdf_deep_analysis'])
            response = invoke_json_llm(
                self.llm,
                prompt.format_messages(
                    target_dataset=dataset_name,
                    full_text=paper_content
                ),
                logger=logger,
                context=f"Paper analysis for {dataset_name}",
            )
            result = self._parse_json_response(response.content, context=f"Paper analysis for {dataset_name}")
            relationships = result.get('source_datasets', [])
            logger.info(
                f"Paper analysis summary for {dataset_name}: "
                f"{len(relationships)} sources -> {summarize_source_relationships(relationships, limit=12)}"
            )
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing paper for datasets: {str(e)}")
            return []
    
    def search_arxiv_by_title(self, paper_title: str) -> Optional[str]:
        """Search arXiv by paper title and return paper URL"""
        try:
            logger.info(f"Searching arXiv for paper: {paper_title}")
            
            clean_title = self._clean_title_for_search(paper_title)
            
            search = arxiv.Search(
                query=f'ti:"{clean_title}"',
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = list(search.results())
            
            if results:
                best_match = results[0]
                
                title_similarity = self._calculate_title_similarity(paper_title, best_match.title)
                
                if title_similarity > 0.6:
                    logger.info(f"Found matching arXiv paper: {best_match.entry_id}")
                    return best_match.entry_id
                else:
                    logger.warning(f"Title similarity too low: {title_similarity}")
            
            keywords = self._extract_keywords_from_title(paper_title)
            if keywords:
                search = arxiv.Search(
                    query=' AND '.join(keywords[:3]),
                    max_results=5,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                results = list(search.results())
                for result in results:
                    title_similarity = self._calculate_title_similarity(paper_title, result.title)
                    if title_similarity > 0.5:
                        logger.info(f"Found paper via keywords: {result.entry_id}")
                        return result.entry_id
            
            logger.warning(f"No matching arXiv paper found for: {paper_title}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching arXiv for title '{paper_title}': {str(e)}")
            return None
    
    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for search"""
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
        title = re.sub(r'[{}\\]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def _extract_keywords_from_title(self, title: str) -> List[str]:
        """Extract keywords from title"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\w+', title.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:5]
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity using Jaccard similarity"""
        words1 = set(re.findall(r'\w+', title1.lower()))
        words2 = set(re.findall(r'\w+', title2.lower()))
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _convert_hf_papers_url_to_arxiv(self, paper_url: str) -> str:
        """
        Convert HuggingFace papers URL to arXiv URL
        
        Example: https://huggingface.co/papers/2503.16212 -> https://arxiv.org/pdf/2503.16212
        
        Args:
            paper_url: Paper URL
            
        Returns:
            Converted URL (original URL if not HF papers format)
        """
        hf_papers_pattern = r'https?://huggingface\.co/papers/([0-9]{4}\.[0-9]{4,5})'
        match = re.search(hf_papers_pattern, paper_url)
        
        if match:
            arxiv_id = match.group(1)
            arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
            logger.info(f"Converted URL: {paper_url} -> {arxiv_url}")
            return arxiv_url
        
        return paper_url
    
    def _extract_arxiv_id(self, arxiv_url: str) -> Optional[str]:
        """Extract arXiv ID from URL"""
        patterns = [
            r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})',
            r'arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, arxiv_url)
            if match:
                return match.group(1)
        return None
    
    def _parse_json_response(self, response_text: str, context: str = "PaperAgent") -> Dict[str, Any]:
        """Parse LLM JSON response with repair and salvage."""
        return parse_llm_json_response(response_text, logger=logger, context=context)
