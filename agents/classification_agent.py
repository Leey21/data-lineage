import logging
from typing import Set, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models import DataCategory, DataLineageState
from ..utils import invoke_json_llm, parse_llm_json_response

logger = logging.getLogger(__name__)


def _deduplicate_evidence(evidence_list: List[str]) -> List[str]:
    """Deduplicate evidence strings while preserving order. Empty strings are skipped."""
    seen = set()
    result = []
    for ev in evidence_list:
        ev = (ev or "").strip()
        if not ev:
            continue
        if ev not in seen:
            seen.add(ev)
            result.append(ev)
    return result


class ClassificationAgent:
    """Dataset classification agent"""

    def __init__(self, llm: ChatOpenAI, enable_multimodal: bool = False):
        self.llm = llm
        self.enable_multimodal = enable_multimodal

    def classify_dataset(
        self,
        dataset_name: str,
        description: str,
        source_names: Optional[List[str]] = None,
        evidence_list: Optional[List[str]] = None,
    ) -> Set[DataCategory]:
        from ..prompts import PROMPTS
        prompt_key = 'dataset_classification_multimodal' if self.enable_multimodal else 'dataset_classification_detailed'
        prompt_template = PROMPTS[prompt_key]
        prompt = ChatPromptTemplate.from_template(prompt_template)

        source_names = source_names or []
        evidence_list = evidence_list or []
        unique_evidence = _deduplicate_evidence(evidence_list)

        source_section = ", ".join(source_names) if source_names else "Not available"
        if unique_evidence:
            evidence_section = "\n".join(f"{i+1}. {ev}" for i, ev in enumerate(unique_evidence))
        else:
            evidence_section = "Not available"

        try:
            response = invoke_json_llm(
                self.llm,
                prompt.format_messages(
                    dataset_name=dataset_name,
                    description=description,
                    source_data_names=source_section,
                    evidence_section=evidence_section,
                ),
                logger=logger,
                context=f"Classification for {dataset_name}",
            )
            result = parse_llm_json_response(
                response.content,
                logger=logger,
                context=f"Classification for {dataset_name}",
            )
            
            categories = set()
            for cat_name in result.get('categories', []):
                cat_str = (cat_name or "").strip()
                if not cat_str:
                    continue
                try:
                    categories.add(DataCategory(cat_str))
                except ValueError:
                    logger.warning(f"Invalid category: {cat_name}")
            
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')
            
            logger.info(f"Refined classification for {dataset_name}: {[cat.value for cat in categories]} (confidence: {confidence})")
            if reasoning:
                logger.info(f"Reasoning: {reasoning}")
            
            return categories if categories else {DataCategory.GENERAL}
            
        except Exception as e:
            logger.error(f"Error classifying dataset {dataset_name}: {str(e)}")
            return {DataCategory.GENERAL}
