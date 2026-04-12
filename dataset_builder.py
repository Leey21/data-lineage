"""
DatasetInfo builder - constructs DatasetInfo from basic info and validated relationships.
"""
import logging
import re
from typing import Dict, List, Optional, Any

from langchain_core.prompts import ChatPromptTemplate

from .models import DatasetInfo, DatasetExistence
from .prompts import PROMPTS
from .utils import extract_year_month_from_paper_links as _extract_paper_year_month

logger = logging.getLogger(__name__)


def build_dataset_info(
    name: str,
    basic_info: Dict[str, Any],
    source_relationships: List[Dict],
    api_client: Any,
    llm: Any,
) -> DatasetInfo:
    """
    Build DatasetInfo from basic info and validated source relationships.

    Args:
        name: Dataset name
        basic_info: Basic info from SourcingAgent (readme_content, paper_links, hf_year, etc.)
        source_relationships: Validated relationships from AggregationAgent
        api_client: Object with _get_dataset_info_via_api, _extract_year_from_paper_links
        llm: LLM for summary and data type determination

    Returns:
        DatasetInfo instance
    """
    hf_year = basic_info.get('hf_year')
    description = basic_info.get('readme_content', '')
    source_datasets = [r['name'] for r in source_relationships]

    summary = _summarize_readme(description, name, llm) if description else ""
    data_type = _determine_data_type(description, name, llm) if description else ""

    downloads = None
    try:
        api_info = api_client._get_dataset_info_via_api(name)
        if api_info:
            downloads = api_info.get('downloads') or api_info.get('downloads_count')
            if downloads is not None:
                try:
                    downloads = int(downloads)
                except (ValueError, TypeError):
                    downloads = None
    except Exception as e:
        logger.debug(f"Failed to get downloads count for {name}: {str(e)}")

    paper_year = None
    paper_links = basic_info.get('paper_links', [])
    if paper_links:
        paper_year = _extract_year_from_paper_links(paper_links)
    paper_url = paper_links[0] if paper_links else None

    is_terminal = False
    terminal_reason = ""

    earliest_year = None
    if hf_year and paper_year:
        years = [hf_year, paper_year]
        years.sort()
        earliest_year = years[0]
    elif hf_year:
        earliest_year = hf_year
    elif paper_year:
        earliest_year = paper_year

    if earliest_year:
        try:
            year_str = earliest_year.split('-')[0] if '-' in earliest_year else earliest_year
            year = int(year_str)
            if year < 2020:
                is_terminal = True
                terminal_reason = f"Dataset created before or in 2020 (year: {year})"
        except (ValueError, TypeError):
            pass

    if not is_terminal and not source_datasets:
        is_terminal = True
        terminal_reason = "No source datasets found"

    return DatasetInfo(
        name=name,
        hf_year=hf_year,
        paper_year=paper_year,
        description=description,
        summary=summary,
        data_type=data_type,
        source_datasets=source_datasets,
        source_datasets_with_relationships=source_relationships,
        exists_on_hf=DatasetExistence.EXISTS,
        is_terminal_node=is_terminal,
        terminal_reason=terminal_reason,
        paper_url=paper_url,
        downloads=downloads,
    )


def _summarize_readme(readme_content: str, dataset_name: str, llm: Any) -> str:
    """Generate concise summary of README using LLM."""
    if not readme_content or len(readme_content.strip()) < 50:
        return ""
    try:
        content = readme_content[:2000] if len(readme_content) > 2000 else readme_content
        prompt = ChatPromptTemplate.from_template(PROMPTS['readme_summary'])
        response = llm.invoke(
            prompt.format_messages(dataset_name=dataset_name, content=content)
        )
        summary = response.content.strip()
        summary = re.sub(r'\s+', ' ', summary)
        word_count = len(summary.split())
        if word_count > 20:
            summary = ' '.join(summary.split()[:20])
        return summary
    except Exception as e:
        logger.warning(f"Failed to generate summary {dataset_name}: {str(e)}")
        return ""


def _determine_data_type(readme_content: str, dataset_name: str, llm: Any) -> str:
    """Determine if dataset is synthetic or manual using LLM."""
    if not readme_content or len(readme_content.strip()) < 50:
        return "manual"
    try:
        content = readme_content[:2000] if len(readme_content) > 2000 else readme_content
        prompt = ChatPromptTemplate.from_template(PROMPTS['dataset_type_determination'])
        response = llm.invoke(
            prompt.format_messages(dataset_name=dataset_name, content=content)
        )
        result = response.content.strip().lower()
        if 'synthetic' in result:
            return 'synthetic'
        if 'manual' in result:
            return 'manual'
        content_lower = readme_content.lower()
        synthetic_keywords = ['synthetic', 'generated', 'artificial', 'simulation']
        manual_keywords = ['collected', 'human', 'real-world', 'natural', 'scraped']
        synthetic_score = sum(1 for k in synthetic_keywords if k in content_lower)
        manual_score = sum(1 for k in manual_keywords if k in content_lower)
        return 'synthetic' if synthetic_score > manual_score else 'manual'
    except Exception as e:
        logger.warning(f"Failed to determine data type {dataset_name}: {str(e)}")
        return "manual"


def _extract_year_from_paper_links(paper_links: List[str]) -> Optional[str]:
    """Extract earliest year-month from paper links."""
    return _extract_paper_year_month(paper_links)


def _extract_year_from_arxiv_url(arxiv_url: str) -> Optional[str]:
    """Extract YYYY-MM from an arXiv URL using the YYMM.NNNNN identifier format."""
    try:
        match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d+', arxiv_url)
        if match:
            yy, month = int(match.group(1)), int(match.group(2))
            year = 1900 + yy if yy >= 91 else 2000 + yy
            if 1 <= month <= 12:
                return f"{year}-{month:02d}"
    except Exception:
        pass
    return None
