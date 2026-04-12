import logging
import json
import os
from typing import Dict, List, Optional, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from .models import DataLineageState, DatasetInfo, DataLineage, DatasetExistence
from .agents import PaperAgent, ClassificationAgent
from .agents.sourcing_agent import SourcingAgent
from .agents.extracting_agent import ExtractingAgent
from .agents.tracing_agent import TracingAgent
from .agents.aggregation_agent import (
    DatasetValidator, DatasetDeduplicator, AggregationAgent
)
from .dataset_builder import build_dataset_info
from .utils import (
    extract_year_month_from_paper_links as _extract_paper_year_month,
    summarize_source_relationships,
)

logger = logging.getLogger(__name__)

def _get_recursion_limit(max_depth: Optional[int]) -> int:
    """Get recursion limit"""
    if max_depth is None:
        return 99999999
    else:
        # LangGraph recursion limit counts node transitions across multiple workflow nodes.
        # Use a generous buffer so low max_depth values do not fail early.
        return max(12, max_depth * 12)

class WorkflowState(TypedDict):
    """Workflow state type definition"""
    current_dataset: str
    lineage_state: DataLineageState
    max_depth: int
    current_depth: int
    processed_count: int
    error_message: Optional[str]
    should_continue: bool
    graph_file: Optional[str]
    data_file: Optional[str]

def _get_file_paths(state: WorkflowState, lineage_state: DataLineageState) -> tuple:
    """Get file paths"""
    graph_file = state.get("graph_file") or getattr(lineage_state, 'graph_file', None)
    data_file = state.get("data_file") or getattr(lineage_state, 'data_file', None)
    return graph_file, data_file

def _log_error_and_continue(logger, error_msg: str, state: WorkflowState) -> WorkflowState:
    """Unified error handling and logging"""
    logger.error(error_msg)
    state["error_message"] = error_msg
    state["should_continue"] = True
    return state


def _log_stage_relationships(stage_label: str, relationships: List[Dict[str, Any]]) -> None:
    """Log concise source summary at INFO and detailed payload at DEBUG."""
    logger.info(
        f"{stage_label}: {len(relationships)} sources -> "
        f"{summarize_source_relationships(relationships, limit=12)}"
    )
    if relationships:
        logger.debug(f"{stage_label} details: {json.dumps(relationships, ensure_ascii=False)}")

class DataLineageWorkflow:
    """Data lineage analysis workflow"""
    
    def __init__(self, llms: Dict[str, Any], max_depth: Optional[int] = 10, api_token: str = None, enable_blog_analysis: bool = True, enable_paper_analysis: bool = True, enable_multimodal: bool = False, enable_pdf_crop: bool = False):
        """
        Initialize workflow.

        Args:
            llms: Dict with keys: sourcing, tracing, paper, classification, aggregation, dataset_builder
            max_depth: Maximum recursion depth, None for unlimited
            api_token: HuggingFace API token
            enable_blog_analysis: Enable blog analysis
            enable_paper_analysis: Enable paper analysis
            enable_multimodal: Use multimodal classification prompt (General/Reasoning/Spatial/Infographic)
            enable_pdf_crop: If True, crop PDF by TOC (intro~conclusion); if False, use full PDF (default)
        """
        self.llm_sourcing = llms.get("sourcing")
        self.llm_tracing = llms.get("tracing")
        self.llm_paper = llms.get("paper")
        self.llm_classification = llms.get("classification")
        self.llm_aggregation = llms.get("aggregation")
        self.llm_dataset_builder = llms.get("dataset_builder")

        self.max_depth = max_depth
        self.enable_paper_analysis = enable_paper_analysis
        self.enable_multimodal = enable_multimodal
        self.enable_pdf_crop = enable_pdf_crop

        self.sourcing_agent = SourcingAgent(
            self.llm_sourcing,
            api_token=api_token,
            enable_blog_analysis=enable_blog_analysis,
            enable_paper_analysis=enable_paper_analysis,
        )
        self.extracting_agent = ExtractingAgent(api_token=api_token)
        self.paper_agent = PaperAgent(self.llm_paper, pdf_crop=enable_pdf_crop)
        self.tracing_agent = TracingAgent(
            self.llm_tracing,
            self.extracting_agent,
            enable_blog_analysis=enable_blog_analysis,
            enable_paper_analysis=enable_paper_analysis,
            pdf_crop=enable_pdf_crop,
            paper_agent=self.paper_agent,
        )
        self.classification_agent = ClassificationAgent(self.llm_classification, enable_multimodal=enable_multimodal)

        self.validator = DatasetValidator(self.sourcing_agent, self.llm_aggregation)
        self.deduplicator = DatasetDeduplicator()
        self.aggregator = AggregationAgent(self.validator, self.deduplicator)
        
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create workflow graph"""
        
        workflow = StateGraph(WorkflowState)
        
        workflow.add_node("process_dataset", self._process_dataset_node)
        workflow.add_node("add_to_queue", self._add_to_queue_node)
        workflow.add_node("get_next_dataset", self._get_next_dataset_node)
        workflow.add_node("finalize", self._finalize_node)
        
        workflow.set_entry_point("process_dataset")
        
        workflow.add_edge("process_dataset", "add_to_queue")
        workflow.add_edge("add_to_queue", "get_next_dataset")
        
        workflow.add_conditional_edges(
            "get_next_dataset",
            self._should_continue,
            {
                True: "process_dataset",
                False: "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)

        return workflow.compile(debug=False)
    
    def _check_early_skip(self, dataset_name: str, lineage_state: DataLineageState) -> Optional[Dict]:
        """Check if dataset analysis should be skipped early based on earliest year"""
        try:
            basic_info = self.sourcing_agent._get_dataset_info_via_api(dataset_name)
            if not basic_info:
                return None
            
            hf_year_month = self.sourcing_agent._extract_year_from_api_info(basic_info)
            
            paper_year_month = None
            paper_url = None
            try:
                readme_content = self.sourcing_agent._get_readme_via_api(dataset_name)
                if readme_content:
                    paper_links = self.sourcing_agent._extract_paper_links(readme_content)
                    if paper_links:
                        paper_url = paper_links[0]
                        paper_year_month = self._extract_year_from_paper_links(paper_links)
            except Exception as e:
                pass
            
            earliest_year_month = None
            if hf_year_month and paper_year_month:
                years = [hf_year_month, paper_year_month]
                years.sort()
                earliest_year_month = years[0]
            elif hf_year_month:
                earliest_year_month = hf_year_month
            elif paper_year_month:
                earliest_year_month = paper_year_month
            else:
                return None
            
            year = None
            if isinstance(earliest_year_month, str):
                try:
                    year = int(earliest_year_month.split('-')[0])
                except (ValueError, IndexError):
                    return None
            elif isinstance(earliest_year_month, int):
                year = earliest_year_month
            
            hf_time_str = hf_year_month if hf_year_month else "N/A"
            paper_time_str = paper_year_month if paper_year_month else "N/A"
            earliest_time_str = earliest_year_month if earliest_year_month else "N/A"
            
            if year and year < 2020:
                logger.info(f"Early skip: {dataset_name} - HF={hf_time_str}, Paper={paper_time_str}, Earliest={earliest_time_str} (year={year}<=2020)")
                downloads = None
                try:
                    if basic_info:
                        downloads = basic_info.get('downloads') or basic_info.get('downloads_count')
                        if downloads is not None:
                            try:
                                downloads = int(downloads)
                            except (ValueError, TypeError):
                                downloads = None
                except Exception:
                    pass
                
                dataset_info = DatasetInfo(
                    name=dataset_name,
                    hf_year=hf_year_month,
                    paper_year=paper_year_month,
                    description="Early skip - dataset created before or in 2020",
                    source_datasets=[],
                    source_datasets_with_relationships=[],
                    exists_on_hf=DatasetExistence.EXISTS,
                    is_terminal_node=True,
                    terminal_reason=f"Dataset created before or in 2020 (year: {year})",
                    paper_url=paper_url,
                    downloads=downloads
                )
                
                lineage_state.add_dataset(dataset_info)
                
                return {
                    "reason": f"Dataset created before or in 2020 (year: {year})",
                    "dataset_info": dataset_info
                }
            
            if year:
                logger.info(f"Early skip check: {dataset_name} - HF={hf_time_str}, Paper={paper_time_str}, Earliest={earliest_time_str} (year={year}>2020), continuing")
            return None
            
        except Exception as e:
            logger.warning(f"Error in early skip check for {dataset_name}: {str(e)}")
            return None
    
    def _process_dataset_node(self, state: WorkflowState) -> WorkflowState:
        """Process current dataset node: Sourcing -> Extracting -> Tracing -> Aggregation"""
        try:
            current_dataset = state["current_dataset"]
            lineage_state = state["lineage_state"]
            
            logger.info(f"Processing dataset: {current_dataset} (depth: {state['current_depth']})")

            if lineage_state.is_processed(current_dataset):
                logger.info(f"[SKIP] Dataset {current_dataset} already processed, skipping")
                state["should_continue"] = True
                return state
            
            from .models import DatasetExistence
            existence = self.validator.verify_existence(current_dataset)
            if existence == DatasetExistence.NOT_FOUND:
                logger.info(f"Dataset {current_dataset} not found on HuggingFace, skipping")
                lineage_state.mark_processed(current_dataset)
                state["should_continue"] = True
                return state
            elif existence == DatasetExistence.UNCERTAIN:
                logger.warning(f"Dataset {current_dataset} existence uncertain, proceeding with caution")
            
            early_skip_result = self._check_early_skip(current_dataset, lineage_state)
            if early_skip_result:
                logger.info(f"Early skip for {current_dataset}: {early_skip_result['reason']}")
                lineage_state.mark_processed(current_dataset)
                self._immediate_write_dataset_results(lineage_state, current_dataset, state)
                state["should_continue"] = True
                return state
            
            dataset_info = self._process_dataset_with_new_agents(current_dataset)
        
            if dataset_info is None:
                logger.error(f"Failed to process dataset: {current_dataset}")
                lineage_state.mark_error(current_dataset)
                state["error_message"] = f"Failed to process dataset: {current_dataset}"
                state["should_continue"] = True
                return state
            
            lineage_state.add_dataset(dataset_info)

            source_names = [r['name'] for r in dataset_info.source_datasets_with_relationships]
            evidence_list = [
                r.get('evidence', '') or ''
                for r in dataset_info.source_datasets_with_relationships
            ]
            classification_description = dataset_info.summary or dataset_info.description
            if classification_description and len(classification_description) > 2000:
                classification_description = classification_description[:2000]
            final_categories = self.classification_agent.classify_dataset(
                dataset_name=current_dataset,
                description=classification_description,
                source_names=source_names,
                evidence_list=evidence_list,
            )
            dataset_info.categories = final_categories

            for relationship in dataset_info.source_datasets_with_relationships:
                source_dataset = relationship['name']
                if not self._relationship_exists(lineage_state, current_dataset, source_dataset) and current_dataset != source_dataset:
                    if not self._has_reverse_relationship(lineage_state, current_dataset, source_dataset):

                        meta_info_dict = {
                            "source": relationship.get("source"),
                            "confidence": relationship.get("confidence"),
                            "relationship": relationship.get("relationship"),
                            "evidence": relationship.get("evidence"),
                            "exists": True if relationship.get("exists_on_hf") == DatasetExistence.EXISTS else False
                        }
                        from .models import DataLineage
                        lineage = DataLineage(
                            target=current_dataset,
                            source=source_dataset,
                            meta_info=json.dumps(meta_info_dict, ensure_ascii=False)
                        )
                        lineage_state.add_lineage(lineage)
                        logger.info(f"Added lineage: {current_dataset} ← {source_dataset}")
                    else:
                        logger.warning(f"Skipping reverse relationship: {current_dataset} ← {source_dataset}")
                else:
                    logger.info(f"Relationship already exists: {current_dataset} ← {source_dataset}")

        
            lineage_state.mark_processed(current_dataset)
            
            self._immediate_write_dataset_results(lineage_state, current_dataset, state)
            
            state["processed_count"] += 1
            
            logger.info(f"Completed processing for {current_dataset}")
            
            return state
            
        except Exception as e:
            return _log_error_and_continue(logger, f"Error in process_dataset_node: {str(e)}", state)
    
    def _immediate_write_dataset_results(self, lineage_state: DataLineageState, current_dataset: str, state: WorkflowState) -> None:
        """Write dataset processing results immediately"""
        try:
            graph_file, data_file = _get_file_paths(state, lineage_state)
            
            logger.info(f"Writing immediate results for {current_dataset}")
            
            dataset_lineages = [lineage for lineage in lineage_state.lineages if lineage.target == current_dataset]
            
            dataset_info = lineage_state.datasets.get(current_dataset)
            
            if graph_file and dataset_lineages:
                try:
                    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
                    
                    existing_relations = set()
                    if os.path.exists(graph_file):
                        try:
                            with open(graph_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        data = json.loads(line.strip())
                                        existing_relations.add((data['target'], data['source']))
                        except (json.JSONDecodeError, KeyError, OSError):
                            pass
                    
                    new_lineages = []
                    for lineage in dataset_lineages:
                        if (lineage.target, lineage.source) not in existing_relations:
                            new_lineages.append(lineage)
                    
                    if new_lineages:
                        with open(graph_file, 'a', encoding='utf-8') as f:
                            for lineage in new_lineages:
                                lineage_dict = {
                                    "target": lineage.target,
                                    "source": lineage.source,
                                    "meta_info": lineage.meta_info
                                }
                                f.write(json.dumps(lineage_dict, ensure_ascii=False) + '\n')
                                f.flush()
                                try:
                                    os.fsync(f.fileno())
                                except (AttributeError, OSError):
                                    pass
                        logger.info(f"Wrote {len(new_lineages)} lineage relations for {current_dataset}")
                    else:
                        logger.info(f"No new lineage relations for {current_dataset}")
                        
                except Exception as e:
                    logger.error(f"Error writing lineage data: {str(e)}")
            
            if data_file and dataset_info:
                try:
                    os.makedirs(os.path.dirname(data_file), exist_ok=True)
                    
                    existing_datasets = set()
                    if os.path.exists(data_file):
                        try:
                            with open(data_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        data = json.loads(line.strip())
                                        existing_datasets.add(data['name'])
                        except (json.JSONDecodeError, KeyError, OSError):
                            pass
                    
                    if current_dataset not in existing_datasets:
                        with open(data_file, 'a', encoding='utf-8') as f:
                            dataset_dict = dataset_info.to_dict()
                            f.write(json.dumps(dataset_dict, ensure_ascii=False) + '\n')
                            f.flush()
                            try:
                                os.fsync(f.fileno())
                            except (AttributeError, OSError):
                                pass
                        logger.info(f"Wrote dataset info for {current_dataset}")
                    else:
                        logger.info(f"Dataset {current_dataset} already exists in file")
                        
                except Exception as e:
                    logger.error(f"Error writing dataset data: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error in _immediate_write_dataset_results: {str(e)}")
    
    def _add_to_queue_node(self, state: WorkflowState) -> WorkflowState:
        """Add source datasets to queue node"""
        try:
            current_dataset = state["current_dataset"]
            lineage_state = state["lineage_state"]
            current_depth = state["current_depth"]
            max_depth = state["max_depth"]
            
            dataset_info = lineage_state.datasets.get(current_dataset)
            
            should_add_to_queue = dataset_info and (max_depth is None or current_depth < max_depth)
            
            if should_add_to_queue and not dataset_info.is_terminal_node:
                added_count = 0
                for rel in dataset_info.source_datasets_with_relationships:
                    source_name = rel.get('name')
                    if not source_name:
                        continue
                    if rel.get('exists_on_hf') != DatasetExistence.EXISTS:
                        continue
                    if not lineage_state.is_processed(source_name):
                        lineage_state.add_to_queue(source_name)
                        added_count += 1
                logger.info(f"Added {added_count} source datasets (on HF) to queue from {current_dataset}")
            else:
                if dataset_info and dataset_info.is_terminal_node:
                    logger.info(f"Dataset {current_dataset} is terminal node, not adding to queue")
                elif max_depth is not None and current_depth >= max_depth:
                    logger.info(f"Reached maximum depth {max_depth}, not adding more datasets to queue")
                elif not dataset_info:
                    logger.info(f"No dataset info found for {current_dataset}")
                else:
                    logger.info(f"No more datasets to add to queue from {current_dataset}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in add_to_queue_node: {str(e)}")
            return state
    
    def _get_next_dataset_node(self, state: WorkflowState) -> WorkflowState:
        """Get next dataset node"""
        try:
            lineage_state = state["lineage_state"]
            current_depth = state["current_depth"]
            max_depth = state["max_depth"]
            
            if max_depth is not None and current_depth >= max_depth:
                logger.info(f"Reached maximum depth {max_depth}, workflow will end")
                state["should_continue"] = False
                return state
            
            next_dataset = lineage_state.get_next_from_queue()
            
            if next_dataset:
                if not lineage_state.should_process(next_dataset):
                    logger.info(f"Skipping dataset {next_dataset}: already processed or in error list")
                    return self._get_next_dataset_node(state)
                
                from .models import DatasetExistence
                existence = self.validator.verify_existence(next_dataset)
                if existence == DatasetExistence.NOT_FOUND:
                    logger.info(f"Dataset {next_dataset} not found on HuggingFace, skipping")
                    lineage_state.mark_processed(next_dataset)
                    return self._get_next_dataset_node(state)
                elif existence == DatasetExistence.UNCERTAIN:
                    logger.warning(f"Dataset {next_dataset} existence uncertain, proceeding with caution")
                
                dataset_info = lineage_state.datasets.get(next_dataset)
                if dataset_info and dataset_info.is_terminal_node:
                    logger.info(f"Dataset {next_dataset} is terminal: {dataset_info.terminal_reason}")
                    lineage_state.mark_processed(next_dataset)
                    return self._get_next_dataset_node(state)
                
                state["current_dataset"] = next_dataset
                state["current_depth"] += 1
                state["should_continue"] = True
                if "graph_file" not in state:
                    state["graph_file"] = getattr(lineage_state, 'graph_file', None)
                if "data_file" not in state:
                    state["data_file"] = getattr(lineage_state, 'data_file', None)
                if max_depth is None:
                    logger.info(f"Next dataset to process: {next_dataset} (depth: {state['current_depth']}, unlimited)")
                else:
                    logger.info(f"Next dataset to process: {next_dataset} (depth: {state['current_depth']})")
            else:
                state["should_continue"] = False
                logger.info("No more datasets in queue, workflow will end")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in get_next_dataset_node: {str(e)}")
            state["should_continue"] = False
            return state
    
    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize node"""
        logger.info("Data lineage workflow completed!")
        summary = state["lineage_state"].get_state_summary()
        logger.info(f"Final summary: {summary}")
        return state
    
    def _should_continue(self, state: WorkflowState) -> bool:
        """Determine if processing should continue"""
        should_continue = state.get("should_continue", False)
        current_depth = state.get("current_depth", 0) 
        max_depth = state.get("max_depth")
        
        if max_depth is not None and current_depth >= max_depth:
            logger.info(f"Stopping workflow: reached maximum depth {max_depth}")
            return False
            
        return should_continue
    
    def run_single_dataset_with_shared_state(
        self, 
        dataset_name: str, 
        shared_state: DataLineageState,
        graph_file: str = None,
        data_file: str = None
    ) -> DataLineageState:
        """
        Run single dataset analysis with shared state
        
        Args:
            dataset_name: Dataset name to analyze
            shared_state: Shared data lineage state object
            graph_file: Lineage relations file path for immediate write
            data_file: Dataset info file path for immediate write
        
        Returns:
            Updated data lineage state object
        """
        logger.info(f"Starting single dataset analysis with shared state: {dataset_name}")
        
        initial_state: WorkflowState = {
            "current_dataset": dataset_name,
            "lineage_state": shared_state,
            "max_depth": self.max_depth,
            "current_depth": 0,
            "processed_count": 0,
            "error_message": None,
            "should_continue": True,
            "graph_file": graph_file,
            "data_file": data_file
        }
        
        try:
            recursion_limit = _get_recursion_limit(self.max_depth)
            final_state = self.workflow.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit}
            )
        except Exception as e:
            logger.error(f"Workflow execution failed for {dataset_name}: {str(e)}")
            return shared_state
        
        result_state = final_state["lineage_state"]
        
        return result_state

    def _has_reverse_relationship(self, lineage_state: DataLineageState, target: str, source: str) -> bool:

        for lineage in lineage_state.lineages:
            if lineage.target == source and lineage.source == target:
                return True
        return False
    
    def _process_dataset_with_new_agents(self, dataset_name: str) -> Optional[DatasetInfo]:
        """
        Process dataset via sequential agent pipeline:

        1. Sourcing: Analyze README, extract resource links (papers, blogs, GitHub)
        2. Extracting: Fetch text content from those links
        3. Tracing: Agent reads text and traces source datasets
        4. Aggregation: Integrate results, filter, resolve names, validate temporal order
        """
        try:
            logger.info(f"Processing dataset with new agent architecture: {dataset_name}")
            
            logger.info("=== Step 1: Sourcing - Extract resource links ===")
            basic_info = self.sourcing_agent.extract_resources(dataset_name)
            if not basic_info:
                logger.warning(f"Failed to extract resources for {dataset_name}")
                return None
            
            logger.info("=== Step 2: Extracting - Fetch content from resources ===")
            extracted = {}
            extracted['readme'] = basic_info.get('readme_content', '')
            extracted['blogs'] = []
            extracted['github'] = {}
            for url in basic_info.get('blog_links', [])[:3]:
                content = self.extracting_agent.extract_blog_content(url)
                if content:
                    extracted['blogs'].append((url, content))
            for url in basic_info.get('github_links', [])[:3]:
                content = self.extracting_agent.extract_github_readme(url)
                if content:
                    extracted['github'][url] = content

            logger.info("=== Step 3: Tracing - Trace source datasets from content ===")
            all_relationships = []
            if extracted['readme']:
                readme_sources = self.tracing_agent.trace_from_readme(extracted['readme'], dataset_name)
                for s in readme_sources:
                    s['source'] = 'HuggingFace README'
                _log_stage_relationships("Tracing/HF README", readme_sources)
                all_relationships.extend(readme_sources)
            for url, paragraphs in extracted['blogs']:
                blog_sources = self.tracing_agent.trace_from_blog_content(paragraphs, dataset_name)
                for s in blog_sources:
                    s['source'] = 'Blog'
                _log_stage_relationships(f"Tracing/Blog {url}", blog_sources)
                all_relationships.extend(blog_sources)
            for url, content in extracted['github'].items():
                github_sources = self.tracing_agent.trace_from_github(url, content, dataset_name)
                for s in github_sources:
                    s['source'] = 'GitHub'
                _log_stage_relationships(f"Tracing/GitHub {url}", github_sources)
                all_relationships.extend(github_sources)
            if self.enable_paper_analysis:
                for paper_url in basic_info.get('paper_links', [])[:2]:
                    paper_sources = self.tracing_agent.trace_from_paper(paper_url, dataset_name)
                    for s in paper_sources:
                        s['source'] = 'Paper'
                    _log_stage_relationships(f"Tracing/Paper {paper_url}", paper_sources)
                    all_relationships.extend(paper_sources)
                if not basic_info.get('paper_links') and basic_info.get('paper_titles'):
                    for title in basic_info.get('paper_titles', [])[:2]:
                        title_sources = self.tracing_agent.trace_from_paper_title(title, dataset_name, basic_info)
                        for s in title_sources:
                            s['source'] = 'Paper'
                        _log_stage_relationships(f"Tracing/PaperTitle {title}", title_sources)
                        all_relationships.extend(title_sources)
            
            logger.info(f"Total source datasets found: {len(all_relationships)}")
            
            logger.info("=== Step 4: Aggregation - Integrate and validate results ===")
            target_year = None
            target_year_candidates = []
            if basic_info.get('hf_year'):
                target_year_candidates.append(basic_info['hf_year'])
            paper_year = self._extract_year_from_paper_links(basic_info.get('paper_links', []))
            if paper_year:
                target_year_candidates.append(paper_year)
            if target_year_candidates:
                target_year_candidates.sort()
                target_year = target_year_candidates[0]
            if not target_year:
                api_info = self.sourcing_agent._get_dataset_info_via_api(dataset_name)
                if api_info:
                    target_year = self.sourcing_agent._extract_year_from_api_info(api_info)
            validated_relationships = self.aggregator.integrate_results(
                target_dataset=dataset_name,
                candidates=all_relationships,
                target_year=target_year,
            )
            return build_dataset_info(
                name=dataset_name,
                basic_info=basic_info,
                source_relationships=validated_relationships,
                api_client=self.sourcing_agent,
                llm=self.llm_dataset_builder,
            )
            
        except Exception as e:
            logger.error(f"Error processing dataset with new agents {dataset_name}: {str(e)}")
            return None
    
    def _relationship_exists(self, lineage_state: DataLineageState, target: str, source: str) -> bool:
        for lineage in lineage_state.lineages:
            if lineage.target == target and lineage.source == source:
                return True
        return False
    
    def _extract_year_from_paper_links(self, paper_links: List[str]) -> Optional[str]:
        """Extract earliest publication year-month from paper links."""
        return _extract_paper_year_month(paper_links)
