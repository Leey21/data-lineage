import logging
from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..prompts import PROMPTS
from ..utils import summarize_source_relationships, invoke_json_llm, parse_llm_json_response
from .extracting_agent import ExtractingAgent
from .paper_agent import PaperAgent

logger = logging.getLogger(__name__)

class TracingAgent:
    """
    Tracing Agent - traces datasets from text content
    
    Responsible for analyzing text content (README, blogs, papers, GitHub) 
    to identify source datasets mentioned in the content.
    """
    
    def __init__(self, llm: ChatOpenAI, extracting_agent: ExtractingAgent, 
                 enable_blog_analysis: bool = True, enable_paper_analysis: bool = True,
                 pdf_crop: bool = False, paper_agent: Optional[PaperAgent] = None):
        self.llm = llm
        self.extracting_agent = extracting_agent
        self.enable_blog_analysis = enable_blog_analysis
        self.enable_paper_analysis = enable_paper_analysis
        self.pdf_crop = pdf_crop
        self._blog_cache = {}
        self.paper_agent = paper_agent or PaperAgent(self.llm, pdf_crop=self.pdf_crop)
    
    def trace_from_readme(self, readme_content: str, target_dataset: str) -> List[Dict]:
        """Trace source datasets from README content"""
        if not readme_content:
            return []
        
        try:
            prompt = ChatPromptTemplate.from_template(PROMPTS['hf_readme_analysis'])
            response = invoke_json_llm(
                self.llm,
                prompt.format_messages(
                    target_dataset=target_dataset,
                    content=readme_content
                ),
                logger=logger,
                context=f"README tracing for {target_dataset}",
            )
            result = self._parse_json_response(response.content, context=f"README tracing for {target_dataset}")
            relationships = result.get('source_datasets', [])
            logger.info(
                f"README tracing for {target_dataset}: "
                f"{len(relationships)} sources -> {summarize_source_relationships(relationships)}"
            )
            return relationships
        except Exception as e:
            logger.error(f"Error tracing from README: {str(e)}")
        return []
    
    def trace_from_blog(self, blog_url: str, target_dataset: str) -> List[Dict]:
        """Trace source datasets from blog URL (fetches content internally)."""
        cache_key = (blog_url, target_dataset)
        if cache_key in self._blog_cache:
            cached = self._blog_cache[cache_key]
            if cached.get('datasets'):
                return cached['datasets']
        paragraphs = self.extracting_agent.extract_blog_content(blog_url)
        if not paragraphs:
            return []
        result = self.trace_from_blog_content(paragraphs, target_dataset)
        self._blog_cache[cache_key] = {'datasets': result}
        return result

    def trace_from_blog_content(self, paragraphs: List[str], target_dataset: str) -> List[Dict]:
        """Trace source datasets from pre-fetched blog paragraphs."""
        if not paragraphs:
            return []
        all_datasets = []
        for i, paragraph in enumerate(paragraphs):
            try:
                logger.debug(f"Analyzing blog paragraph {i+1}/{len(paragraphs)}")
                
                prompt = ChatPromptTemplate.from_template(PROMPTS['blog_content_analysis'])
                response = invoke_json_llm(
                    self.llm,
                    prompt.format_messages(content=paragraph, target_dataset=target_dataset),
                    logger=logger,
                    context=f"Blog tracing for {target_dataset} paragraph {i + 1}",
                )
                
                result = self._parse_json_response(
                    response.content,
                    context=f"Blog tracing for {target_dataset} paragraph {i + 1}",
                )
                source_datasets = result.get('source_datasets', [])
                
                formatted_datasets = []
                for d in source_datasets:
                    if isinstance(d, dict) and d.get('name'):
                        formatted_datasets.append({
                            'name': d.get('name', ''),
                            'source': 'blog',
                            'relationship': d.get('relationship', 'Direct Inclusion/Subset'),
                            'confidence': d.get('confidence', 0.7),
                            'evidence': d.get('evidence', ''),
                            'paragraph_index': i,
                            'paragraph_content': paragraph[:50] + "..." if len(paragraph) > 50 else paragraph
                        })
                
                all_datasets.extend(formatted_datasets)
            except Exception as e:
                logger.error(f"Error analyzing blog paragraph {i+1}: {str(e)}")
                continue
        
        unique_datasets = {}
        for dataset in all_datasets:
            name = dataset.get('name', '').lower().strip()
            if name in unique_datasets:
                existing = unique_datasets[name]
                if dataset.get('confidence', 0) > existing.get('confidence', 0):
                    unique_datasets[name] = dataset
                existing_evidence = existing.get('evidence', '')
                new_evidence = dataset.get('evidence', '')
                if new_evidence and new_evidence not in existing_evidence:
                    existing['evidence'] = f"{existing_evidence}; {new_evidence}"
            else:
                unique_datasets[name] = dataset
        
        relationships = list(unique_datasets.values())
        logger.info(
            f"Blog tracing for {target_dataset}: "
            f"{len(relationships)} sources -> {summarize_source_relationships(relationships)}"
        )
        return relationships

    def trace_from_github(self, github_url: str, github_readme: str, target_dataset: str) -> List[Dict]:
        """Trace source datasets from GitHub README"""
        try:
            prompt = ChatPromptTemplate.from_template(PROMPTS['github_analysis'])
            response = invoke_json_llm(
                self.llm,
                prompt.format_messages(
                    repo_url=github_url,
                    readme_content=github_readme,
                    target_dataset=target_dataset
                ),
                logger=logger,
                context=f"GitHub tracing for {target_dataset}",
            )
            
            result = self._parse_json_response(response.content, context=f"GitHub tracing for {target_dataset}")
            datasets = result.get('source_datasets', result.get('datasets_found', []))
            
            formatted_datasets = []
            for d in datasets:
                if isinstance(d, dict) and d.get('name'):
                    formatted_datasets.append({
                        'name': d.get('name', ''),
                        'source': 'github',
                        'relationship': d.get('relationship', 'Direct Inclusion/Subset'),
                        'evidence': d.get('evidence', ''),
                        'confidence': d.get('confidence', result.get('confidence', 0.5))
                    })
            logger.info(
                f"GitHub tracing for {target_dataset}: "
                f"{len(formatted_datasets)} sources -> {summarize_source_relationships(formatted_datasets)}"
            )
            return formatted_datasets
        except Exception as e:
            logger.error(f"Error tracing from GitHub: {str(e)}")
        return []
    
    def trace_from_paper(self, paper_url: str, target_dataset: str) -> List[Dict]:
        """Trace source datasets from paper content (PaperAgent fetches and analyzes)."""
        try:
            paper_result = self.paper_agent.process_paper(target_dataset, paper_url)
            if isinstance(paper_result, dict):
                relationships = paper_result.get('datasets', [])
                logger.info(
                    f"Paper tracing for {target_dataset}: "
                    f"{len(relationships)} sources -> {summarize_source_relationships(relationships)}"
                )
                return relationships
            return []
        except Exception as e:
            logger.error(f"Error tracing from paper: {str(e)}")
        return []
    
    def trace_from_paper_title(self, paper_title: str, target_dataset: str, basic_info: Dict[str, Any]) -> List[Dict]:
        """Trace source datasets from paper title by searching arXiv"""
        try:
            arxiv_url = self.paper_agent.search_arxiv_by_title(paper_title)
            
            if arxiv_url:
                logger.info(f"Found arXiv URL for '{paper_title}': {arxiv_url}")
                if 'paper_links' not in basic_info:
                    basic_info['paper_links'] = []
                if arxiv_url not in basic_info['paper_links']:
                    basic_info['paper_links'].append(arxiv_url)
                
                return self.trace_from_paper(arxiv_url, target_dataset)
            logger.info(f"No arXiv match found for paper title '{paper_title}'")
            return []
        except Exception as e:
            logger.warning(f"Error processing paper title '{paper_title}': {str(e)}")
        return []
    
    def _parse_json_response(self, response_text: str, context: str = "TracingAgent") -> Dict[str, Any]:
        """Parse LLM JSON response with repair and salvage."""
        return parse_llm_json_response(response_text, logger=logger, context=context)
