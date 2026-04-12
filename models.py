from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any
from enum import Enum
import json
import os

class DataCategory(Enum):
    # Text-only categories
    MATH = "Math"
    CODE = "Code"
    GENERAL = "General"
    SCIENCE = "Science"
    # Multimodal-specific categories
    REASONING = "Reasoning"
    SPATIAL = "Spatial"
    INFOGRAPHIC = "Infographic"

class DatasetExistence(Enum):
    """Dataset existence status"""
    EXISTS = "exists"
    NOT_FOUND = "not_found" 
    UNCERTAIN = "uncertain"

@dataclass
class DatasetInfo:
    name: str
    categories: Set[DataCategory] = field(default_factory=set)
    hf_year: Optional[str] = None
    paper_year: Optional[str] = None
    source_datasets: List[str] = field(default_factory=list)
    source_datasets_with_relationships: List[Dict[str, str]] = field(default_factory=list)
    description: str = ""
    summary: str = ""
    data_type: str = ""
    exists_on_hf: Optional[DatasetExistence] = None
    is_terminal_node: bool = False
    terminal_reason: Optional[str] = None
    paper_url: Optional[str] = None
    downloads: Optional[int] = None

    def to_dict(self):
        final_year = self._get_earliest_year()

        return {
            "name": self.name,
            "categories": [cat.value for cat in self.categories],
            "hf_year": self.hf_year,
            "paper_year": self.paper_year,
            "year": final_year,
            "description": self.description,
            "summary": self.summary,
            "data_type": self.data_type,
            "exists_on_hf": self.exists_on_hf.value if self.exists_on_hf else None,
            "is_terminal_node": self.is_terminal_node,
            "terminal_reason": self.terminal_reason,
            "paper_url": self.paper_url,
            "downloads": self.downloads,
        }
    
    def _get_earliest_year(self) -> Optional[str]:
        """Get earliest year from hf_year and paper_year"""
        years = []
        if self.hf_year:
            years.append(self.hf_year)
        if self.paper_year:
            years.append(self.paper_year)
        
        if not years:
            return None
        elif len(years) == 1:
            return years[0]
        else:
            years.sort()
            return years[0]

@dataclass
class DataLineage:
    target: str
    source: str
    meta_info: str = ""
    
    def to_dict(self):
        return {
            "target": self.target,
            "source": self.source,
            "meta_info": self.meta_info
        }

class DataLineageState:
    def __init__(self, load_existing: bool = False, graph_file: str = None, data_file: str = None):
        self.datasets: Dict[str, DatasetInfo] = {}
        self.lineages: List[DataLineage] = []
        self.processed_datasets: Set[str] = set()
        self.queue: List[str] = []
        self.error_datasets: Set[str] = set()
        
        self.graph_file = graph_file
        self.data_file = data_file
        
        if load_existing and graph_file and data_file:
            self.load_existing_data(graph_file, data_file)
    
    def add_dataset(self, dataset_info: DatasetInfo):
        self.datasets[dataset_info.name] = dataset_info
    
    def add_lineage(self, lineage: DataLineage):
        self.lineages.append(lineage)
    
    def mark_processed(self, dataset_name: str):
        self.processed_datasets.add(dataset_name)
    
    def is_processed(self, dataset_name: str) -> bool:
        return dataset_name in self.processed_datasets
    
    def add_to_queue(self, dataset_name: str):
        if dataset_name not in self.queue and not self.is_processed(dataset_name):
            self.queue.append(dataset_name)
    
    def get_next_from_queue(self) -> Optional[str]:
        if self.queue:
            return self.queue.pop(0)
        return None
    
    def mark_error(self, dataset_name: str):
        self.error_datasets.add(dataset_name)
    
    def should_process(self, dataset_name: str, year: Optional[int] = None) -> bool:
        if self.is_processed(dataset_name):
            return False
        if dataset_name in self.error_datasets:
            return False
        return True
    
    def load_existing_data(self, graph_filename: str, data_filename: str):
        """Load historical data from existing files"""
        try:
            if os.path.exists(graph_filename):
                try:
                    with open(graph_filename, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                lineage_dict = json.loads(line)
                                lineage = DataLineage(
                                    target=lineage_dict['target'],
                                    source=lineage_dict['source'],
                                    meta_info=lineage_dict.get('meta_info', '')
                                )
                                self.lineages.append(lineage)
                except (PermissionError, OSError, json.JSONDecodeError, KeyError) as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Cannot read existing graph file {graph_filename}: {e}")
            
            if os.path.exists(data_filename):
                try:
                    with open(data_filename, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data_dict = json.loads(line)
                                categories = set()
                                for cat_name in data_dict.get('categories', []):
                                    try:
                                        categories.add(DataCategory(cat_name))
                                    except ValueError:
                                        pass

                                source_relationships = []
                                for rel in data_dict.get('source_datasets_with_relationships', []):
                                    if not isinstance(rel, dict):
                                        continue
                                    restored_rel = rel.copy()
                                    exists_value = restored_rel.get('exists_on_hf')
                                    if exists_value:
                                        try:
                                            restored_rel['exists_on_hf'] = DatasetExistence(exists_value)
                                        except ValueError:
                                            restored_rel['exists_on_hf'] = None
                                    source_relationships.append(restored_rel)

                                exists_on_hf = data_dict.get('exists_on_hf')
                                if exists_on_hf:
                                    try:
                                        exists_on_hf = DatasetExistence(exists_on_hf)
                                    except ValueError:
                                        exists_on_hf = None
                                
                                dataset_info = DatasetInfo(
                                    name=data_dict['name'],
                                    categories=categories,
                                    hf_year=data_dict.get('hf_year'),
                                    paper_year=data_dict.get('paper_year'),
                                    source_datasets=data_dict.get('source_datasets', []),
                                    source_datasets_with_relationships=source_relationships,
                                    description=data_dict.get('description', ''),
                                    summary=data_dict.get('summary', ''),
                                    data_type=data_dict.get('data_type', ''),
                                    exists_on_hf=exists_on_hf,
                                    is_terminal_node=bool(data_dict.get('is_terminal_node', False)),
                                    terminal_reason=data_dict.get('terminal_reason'),
                                    paper_url=data_dict.get('paper_url'),
                                    downloads=data_dict.get('downloads'),
                                )
                                self.datasets[data_dict['name']] = dataset_info
                                self.processed_datasets.add(data_dict['name'])
                except (PermissionError, OSError, json.JSONDecodeError, KeyError) as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Cannot read existing data file {data_filename}: {e}")
                    
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error loading existing data: {e}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "total_datasets": len(self.datasets),
            "processed_datasets": len(self.processed_datasets),
            "total_lineages": len(self.lineages),
            "queue_size": len(self.queue),
            "error_datasets": len(self.error_datasets)
        }
