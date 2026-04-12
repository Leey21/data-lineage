import logging
import os
import re
from typing import Dict, List, Optional, Any

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..prompts import PROMPTS
from ..utils import safe_requests_get, make_text_preview, invoke_json_llm, parse_llm_json_response

logger = logging.getLogger(__name__)

class SourcingAgent:
    """
    Sourcing Agent - extracts resource links from README content
    
    Responsible for discovering and extracting links to papers, blogs, GitHub repos, etc.
    from dataset README files.
    """
    
    def __init__(self, llm: ChatOpenAI, api_token: str = None, enable_blog_analysis: bool = True, enable_paper_analysis: bool = True):
        self.llm = llm
        self.api_base_url = "https://huggingface.co/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.api_token = api_token or os.getenv('HUGGINGFACE_API_TOKEN')
        if self.api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_token}'
            })
            logger.info("Using HuggingFace API token authentication")
        
        proxy_env = {
            'https_proxy': os.getenv('https_proxy'),
            'http_proxy': os.getenv('http_proxy')
        }
        if proxy_env['https_proxy'] or proxy_env['http_proxy']:
            self.session.proxies.update(proxy_env)
        
        self._api_cache = {}
        self.enable_blog_analysis = enable_blog_analysis
        self.enable_paper_analysis = enable_paper_analysis
    
    def extract_resources(self, dataset_name: str) -> Dict[str, Any]:
        """
        Extract resource links from dataset README
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dictionary containing extracted links and basic info
        """
        try:
            logger.info(f"Extracting resources for dataset: {dataset_name}")
            
            api_info = self._get_dataset_info_via_api(dataset_name)
            readme_content = self._get_readme_via_api(dataset_name)
            
            if readme_content:
                marker = "---\n\n#"
                idx = readme_content.find(marker)
                if idx != -1:
                    readme_content = readme_content[idx + len(marker):]
                logger.info(
                    f"README preview for {dataset_name}: "
                    f"{make_text_preview(readme_content, max_len=500)}"
                )
            
            basic_info = {
                "url": f"https://huggingface.co/datasets/{dataset_name}",
                "readme_content": readme_content if readme_content else '',
                "id": api_info.get('id', '') if api_info else '',
                "hf_year": self._extract_year_from_api_info(api_info) if api_info else None,
            }
            
            extracted_links = self._extract_links_from_content(basic_info.get('readme_content', ''), dataset_name)
            basic_info.update(extracted_links)
            
            logger.info(f"Extracted resources for {dataset_name}: "
                       f"papers={basic_info.get('paper_links', [])}, "
                       f"github={basic_info.get('github_links', [])}, "
                       f"blogs={basic_info.get('blog_links', [])}, "
                       f"paper_titles={basic_info.get('paper_titles', [])})")
            return basic_info
            
        except Exception as e:
            logger.error(f"Error extracting resources for {dataset_name}: {str(e)}")
            return {}
    
    def _extract_links_from_content(self, content: str, target_dataset: str = "") -> Dict[str, List[str]]:
        """Extract links from content using LLM. target_dataset: the dataset name these links should describe."""
        if not content:
            return {"paper_links": [], "paper_titles": [], "github_links": [], "blog_links": []}
        
        extracted_info = self._llm_extract_links_and_info(content, target_dataset)
        paper_titles = self._extract_paper_titles_from_bibtex(content)
        paper_url = self._extract_paper_links(content)
        return {
            "paper_links": self._deduplicate_urls(
                [self._normalize_paper_url(url) for url in extracted_info.get('paper_links', []) + paper_url]
            ),
            "paper_titles": paper_titles,
            "github_links": self._deduplicate_urls(extracted_info.get('github_links', [])),
            "blog_links": self._deduplicate_urls(extracted_info.get('blog_links', [])) if self.enable_blog_analysis else []
        }

    def _deduplicate_urls(self, urls: List[str]) -> List[str]:
        ordered = []
        seen = set()
        for url in urls:
            value = (url or '').strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _normalize_paper_url(self, url: str) -> str:
        """Normalize common paper URLs to a stable canonical form for deduplication."""
        value = (url or '').strip().rstrip(').,;')
        if not value:
            return ""

        arxiv_match = re.search(
            r'(?:arxiv\.org/(?:abs|pdf)/|huggingface\.co/papers/)([0-9]{4}\.[0-9]{4,5})',
            value,
            re.IGNORECASE,
        )
        if arxiv_match:
            return f"https://arxiv.org/abs/{arxiv_match.group(1)}"
        return value
    
    def _llm_extract_links_and_info(self, content: str, target_dataset: str) -> Dict[str, List[str]]:
        """Extract dataset-related links using LLM. target_dataset: the dataset name these links should describe."""
        try:
            prompt = ChatPromptTemplate.from_template(PROMPTS['link_extraction'])
            response = invoke_json_llm(
                self.llm,
                prompt.format_messages(content=content, target_dataset=target_dataset or "this dataset"),
                logger=logger,
                context=f"Link extraction for {target_dataset or 'dataset'}",
            )
            result = self._parse_json_response(response.content, context=f"Link extraction for {target_dataset or 'dataset'}")
            return {
                'paper_links': result.get('paper_links', []),
                'github_links': result.get('github_links', []), 
                'blog_links': result.get('blog_links', [])
            }
        except Exception as e:
            logger.error(f"Error in LLM link extraction: {str(e)}")
            return {'paper_links': [], 'github_links': [], 'blog_links': []}
    
    def _extract_paper_titles_from_bibtex(self, text: str) -> List[str]:
        """Extract paper titles from bibtex using regex"""
        paper_titles = []
        try:
            direct_pattern = r'@\w+\{[^@]*?title\s*=\s*\{([^}]+)\}[^@]*?\}'
            direct_matches = re.findall(direct_pattern, text, re.DOTALL | re.IGNORECASE)
            
            for title in direct_matches:
                cleaned_title = self._clean_bibtex_title(title.strip())
                if cleaned_title and len(cleaned_title) > 10:
                    paper_titles.append(cleaned_title)
            
            if not direct_matches:
                bibtex_entries = self._extract_balanced_bibtex_entries(text)
                for entry in bibtex_entries:
                    title_patterns = [
                        r'title\s*=\s*\{([^}]+)\}',
                        r'title\s*=\s*"([^"]+)"',
                    ]
                    for pattern in title_patterns:
                        title_match = re.search(pattern, entry, re.IGNORECASE)
                        if title_match:
                            cleaned_title = self._clean_bibtex_title(title_match.group(1).strip())
                            if cleaned_title and len(cleaned_title) > 10:
                                paper_titles.append(cleaned_title)
                            break
        except Exception as e:
            logger.debug(f"Error extracting paper titles from bibtex: {str(e)}")
        
        return list(set(paper_titles))
    
    def _clean_bibtex_title(self, title: str) -> str:
        """Clean bibtex title, remove LaTeX commands and extra spaces"""
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
        title = re.sub(r'\\[a-zA-Z]+', '', title)
        title = re.sub(r'[{}]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def _extract_balanced_bibtex_entries(self, text: str) -> List[str]:
        """Extract complete balanced brace bibtex entries"""
        entries = []
        pattern = r'@\w+\s*\{'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_pos = match.start()
            brace_count = 0
            current_pos = match.end() - 1
            
            while current_pos < len(text):
                char = text[current_pos]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        entry = text[start_pos:current_pos + 1]
                        entries.append(entry)
                        break
                current_pos += 1
        
        return entries
    
    def _extract_paper_links(self, text: str) -> List[str]:
        """Extract paper links"""
        patterns = [
            r'https?://arxiv\.org/(?:abs/|pdf/)?([0-9]{4}\.[0-9]{4,5})',
            r'https?://huggingface\.co/papers/([0-9]{4}\.[0-9]{4,5})',
            r'https?://aclanthology\.org/[^)\s\*]+(?:\.pdf)?',
            r'https?://openreview\.net/forum\?id=[^\s\*\)]+',
        ]
        
        links = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if 'arxiv' in pattern or 'huggingface\\.co/papers' in pattern:
                links.extend([f"https://arxiv.org/abs/{match}" for match in matches])
            else:
                links.extend(matches)
        
        return self._deduplicate_urls([self._normalize_paper_url(link) for link in links])
    
    def _get_dataset_info_via_api(self, dataset_name: str) -> Optional[Dict]:
        """Get dataset basic information via API"""
        if dataset_name in self._api_cache and 'info' in self._api_cache[dataset_name]:
            return self._api_cache[dataset_name]['info']
        
        try:
            api_url = f"{self.api_base_url}/datasets/{dataset_name}"
            response = safe_requests_get(api_url, timeout=15, max_retries=5, sleep_time=30,
                                       headers=self.session.headers, proxies=self.session.proxies)
            if response.status_code == 200:
                dataset_info = response.json()
                if dataset_name not in self._api_cache:
                    self._api_cache[dataset_name] = {}
                self._api_cache[dataset_name]['info'] = dataset_info
                return dataset_info
        except Exception as e:
            logger.debug(f"API error for {dataset_name}: {str(e)}")
        
        return None
    
    def _get_readme_via_api(self, dataset_name: str) -> Optional[str]:
        """Get README content via API"""
        readme_paths = [
            f"https://huggingface.co/datasets/{dataset_name}/raw/main/README.md",
            f"https://huggingface.co/datasets/{dataset_name}/raw/master/README.md"
        ]
        for path in readme_paths:
            try:
                response = safe_requests_get(path, timeout=30, max_retries=5, sleep_time=30,
                                           headers=self.session.headers, proxies=self.session.proxies)
                if response.status_code == 200 and len(response.text.strip()) > 100:
                    return response.text
            except Exception:
                continue
        return None
    
    def _extract_year_from_api_info(self, api_info: Dict) -> Optional[str]:
        """Extract year-month from API info in YYYY-MM format"""
        try:
            created_at = api_info.get('createdAt')
            if created_at:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                return f"{dt.year}-{dt.month:02d}"
            for field in ['lastModified', 'updatedAt', 'publishedAt']:
                if field in api_info and api_info[field]:
                    try:
                        dt = datetime.fromisoformat(api_info[field].replace('Z', '+00:00'))
                        return f"{dt.year}-{dt.month:02d}"
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Error extracting year from API info: {str(e)}")
        return None
    
    def _parse_json_response(self, response_text: str, context: str = "SourcingAgent") -> Dict[str, Any]:
        """Parse LLM JSON response with repair and salvage."""
        return parse_llm_json_response(response_text, logger=logger, context=context)
