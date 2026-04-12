#!/usr/bin/env python3
"""
Main program for data lineage analysis

Multi-agent collaborative data lineage tracking system implemented with LangChain and LangGraph
"""

import os
import sys
import argparse
import logging
import re
from typing import Optional
from pathlib import Path

from langchain_openai import ChatOpenAI

from .workflow import DataLineageWorkflow
from .models import DataLineageState

PACKAGE_LOGGER_NAME = "data_lineage"
HF_DATASET_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path("output")


class _ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors by log level (console only, no colors in file)."""
    COLORS = {
        logging.DEBUG: "\033[36m",    # cyan
        logging.INFO: "\033[32m",     # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",    # red
        logging.CRITICAL: "\033[35m", # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        plain = super().format(record)
        if not sys.stdout.isatty():
            return plain
        color = self.COLORS.get(record.levelno, "")
        if not color:
            return plain
        idx = plain.find(record.levelname)
        if idx >= 0:
            return plain[:idx] + color + record.levelname + self.RESET + plain[idx + len(record.levelname):]
        return plain


class _FlushFileHandler(logging.FileHandler):
    """File handler that flushes after each emit for real-time log writing."""

    def emit(self, record):
        super().emit(record)
        self.flush()
        try:
            import os
            os.fsync(self.stream.fileno())
        except (AttributeError, OSError):
            pass


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configure package logging with colored console and real-time file output."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(getattr(logging, log_level.upper()))
    package_logger.propagate = False

    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ColoredFormatter(log_format))
    package_logger.addHandler(console)

    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = _FlushFileHandler(log_file, mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter(log_format))
            package_logger.addHandler(fh)
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot write to log file {log_file}: {e}, using console only")


def _is_valid_hf_dataset_name(dataset_name: str) -> bool:
    return bool(HF_DATASET_PATTERN.match((dataset_name or "").strip()))

def validate_environment() -> bool:
    """Validate environment configuration"""
    required = ["OPENAI_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print("Error: Missing required environment variables. Please set:")
        print("  export OPENAI_BASE_URL=\"https://your-api-proxy.com/v1\"")
        print("  export OPENAI_API_KEY=\"sk-your-api-key\"")
        print("  export HUGGINGFACE_API_TOKEN=\"hf_your-token\"  # optional")
        return False
    return True

def _create_llm_instance(model_name: str, base_url: Optional[str] = None) -> ChatOpenAI:
    """Create a single LLM instance. base_url for proxy/relay (e.g., https://your-proxy.com/v1)."""
    kwargs = dict(
        model=model_name,
        temperature=0,
        max_retries=5,
        request_timeout=180,
    )
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return ChatOpenAI(**kwargs)


def create_llms(model_config: dict, base_url: Optional[str] = None) -> dict:
    """
    Create LLM instances for each agent from model config.

    model_config keys: sourcing, tracing, paper, classification, aggregation, dataset_builder
    base_url: API base URL for proxy/relay (applies to all agents if set)
    """
    url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    return {k: _create_llm_instance(v, url) for k, v in model_config.items()}

def _resolve_output_directory(output_dir: str) -> Path:
    """Resolve relative output directories from the package root."""
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = (PACKAGE_DIR / output_path).resolve()
    return output_path


def _setup_output_directory(output_path: Path) -> tuple[Path, bool]:
    """Setup output directory, handle permission issues"""
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path, False
    except PermissionError:
        import tempfile
        fallback_path = Path(tempfile.mkdtemp(prefix="data_lineage_"))
        print(f"Warning: Permission denied for {output_path}, using temporary directory: {fallback_path}")
        return fallback_path, True


def _get_results_directory(output_path: Path, enable_multimodal: bool) -> Path:
    """Choose the modality-specific results directory under the base output path."""
    subdir = "multimodal" if enable_multimodal else "text-only_modality"
    results_path = output_path / subdir
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

def _log_model_configuration(logger, model_config: dict):
    """Log model configuration for each agent"""
    logger.info("Model configuration (per agent):")
    for agent, model in model_config.items():
        logger.info(f"  - {agent}: {model}")

def _log_analysis_parameters(logger, dataset_name: str, max_depth: Optional[int], 
                           enable_blog_analysis: bool, enable_paper_analysis: bool,
                           enable_pdf_crop: bool = False):
    """Log analysis parameters"""
    logger.info(f"Starting analysis for dataset: {dataset_name}")
    if max_depth is None:
        logger.info(f"Max depth: unlimited")
    else:
        logger.info(f"Max depth: {max_depth}")
    logger.info(f"Blog analysis: {'enabled' if enable_blog_analysis else 'disabled'}")
    logger.info(f"Paper analysis: {'enabled' if enable_paper_analysis else 'disabled'}")
    logger.info(f"PDF crop (by TOC): {'enabled' if enable_pdf_crop else 'disabled (full PDF)'}")

def analyze_datasets(
    dataset_file: str,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    max_depth: Optional[int] = None,
    model_config: dict = None,
    base_url: Optional[str] = None,
    log_level: str = "INFO",
    load_existing: bool = True,
    api_token: str = None,
    enable_blog_analysis: bool = True,
    enable_paper_analysis: bool = True,
    enable_multimodal: bool = False,
    enable_pdf_crop: bool = False,
) -> None:

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset file: {str(e)}")
    
    if not dataset_list:
        raise ValueError(f"Dataset file is empty or contains no valid datasets: {dataset_file}")
    
    requested_output_path = _resolve_output_directory(output_dir)
    output_path, used_fallback_output = _setup_output_directory(requested_output_path)
    log_file = str(output_path / "data_lineage.log")
    setup_logging(log_level, log_file)
    logger = logging.getLogger(__name__)
    if used_fallback_output:
        logger.warning(f"Using temporary output directory: {output_path}")
    
    if not validate_environment():
        raise RuntimeError("Environment validation failed")

    if model_config is None:
        model_config = {}
    default_model = model_config.get("default", "gpt-5.4")
    full_config = {
        "sourcing": model_config.get("sourcing", default_model),
        "tracing": model_config.get("tracing", default_model),
        "paper": model_config.get("paper", default_model),
        "classification": model_config.get("classification", default_model),
        "aggregation": model_config.get("aggregation", default_model),
        "dataset_builder": model_config.get("dataset_builder", default_model),
    }
    llms = create_llms(full_config, base_url=base_url)
    _log_model_configuration(logger, full_config)
    if base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"):
        logger.info(f"API base URL: {base_url or os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE')}")
    
    logger.info("=" * 60)
    logger.info("Starting data lineage analysis")
    logger.info(f"Number of datasets to process: {len(dataset_list)}")
    _log_analysis_parameters(logger, "Data lineage analysis", max_depth, enable_blog_analysis, enable_paper_analysis, enable_pdf_crop)
    logger.info("=" * 60)
    
    workflow = DataLineageWorkflow(
        llms=llms,
        max_depth=max_depth,
        api_token=api_token,
        enable_blog_analysis=enable_blog_analysis,
        enable_paper_analysis=enable_paper_analysis,
        enable_multimodal=enable_multimodal,
        enable_pdf_crop=enable_pdf_crop,
    )
    
    results_path = _get_results_directory(output_path, enable_multimodal)
    graph_file = results_path / "graph.jsonl"
    data_file = results_path / "data.jsonl"
    
    try:
        shared_state = DataLineageState(
            load_existing=load_existing,
            graph_file=str(graph_file),
            data_file=str(data_file)
        )
        
        processed_count = 0
        error_count = 0
        
        for i, dataset_name in enumerate(dataset_list, 1):
            try:
                logger.info(f"\n[{i}/{len(dataset_list)}] Processing dataset: {dataset_name}")
                
                if shared_state.is_processed(dataset_name):
                    logger.info(f"[SKIP] Dataset {dataset_name} already processed, skipping")
                    continue

                if not _is_valid_hf_dataset_name(dataset_name):
                    logger.warning(f"Invalid dataset format, expected org/name: {dataset_name}")
                    shared_state.mark_processed(dataset_name)
                    continue
                
                from .models import DatasetExistence
                existence = workflow.validator.verify_existence(dataset_name)
                if existence == DatasetExistence.NOT_FOUND:
                    logger.info(f"Dataset {dataset_name} not found on HuggingFace, skipping")
                    shared_state.mark_processed(dataset_name)
                    continue
                elif existence == DatasetExistence.UNCERTAIN:
                    logger.warning(f"Dataset {dataset_name} existence uncertain, proceeding")
                
                result_state = workflow.run_single_dataset_with_shared_state(
                    dataset_name=dataset_name,
                    shared_state=shared_state,
                    graph_file=str(graph_file),
                    data_file=str(data_file)
                )
                
                shared_state = result_state
                processed_count += 1
                
                summary = shared_state.get_state_summary()
                logger.info(f"Completed processing: {dataset_name}")
                logger.info(f"   Current totals - Datasets: {summary['total_datasets']}, Lineages: {summary['total_lineages']}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {dataset_name}: {str(e)}")
                continue
        
        final_summary = shared_state.get_state_summary()
        logger.info("\n" + "=" * 60)
        logger.info("Data lineage analysis completed!")
        logger.info(f"Total processed datasets: {processed_count}/{len(dataset_list)}")
        logger.info(f"Error datasets: {error_count}")
        logger.info(f"Final statistics:")
        logger.info(f"  - Total datasets: {final_summary['total_datasets']}")
        logger.info(f"  - Processed datasets: {final_summary['processed_datasets']}")
        logger.info(f"  - Lineage relations: {final_summary['total_lineages']}")
        logger.info(f"  - Error datasets: {final_summary['error_datasets']}")
        logger.info("Output files:")
        logger.info(f"  - Lineage relations: {graph_file}")
        logger.info(f"  - Dataset info: {data_file}")
        logger.info(f"  - Log file: {log_file}")
        logger.info("=" * 60)
        
        return None
            
    except Exception as e:
        logger.error(f"Data lineage analysis failed: {str(e)}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Data lineage analysis tool")
    parser.add_argument("dataset_file", help="Path to file containing dataset list (one per line)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--max-depth", type=int, default=-1, help="Maximum recursion depth, -1 for unlimited (default: -1, trace to bottom)")
    parser.add_argument("--model", default="gpt-5.4", help="Default model for all agents")
    parser.add_argument("--model-sourcing", help="Model for SourcingAgent (link extraction from README)")
    parser.add_argument("--model-tracing", help="Model for TracingAgent (trace datasets from README/blog/GitHub)")
    parser.add_argument("--model-paper", help="Model for PaperAgent (paper content analysis)")
    parser.add_argument("--model-classification", help="Model for ClassificationAgent (dataset classification)")
    parser.add_argument("--model-aggregation", help="Model for AggregationAgent (infer missing dataset names)")
    parser.add_argument("--model-dataset-builder", help="Model for DatasetBuilder (README summary, data type)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--no-load-existing", action="store_true", help="Do not load existing data, start fresh")
    parser.add_argument("--blog-analysis", action="store_true", default=True, help="Enable blog analysis (default: True)")
    parser.add_argument("--no-blog-analysis", action="store_true", help="Disable blog analysis")
    parser.add_argument("--paper-analysis", action="store_true", default=True, help="Enable paper analysis (default: True)")
    parser.add_argument("--no-paper-analysis", action="store_true", help="Disable paper analysis")
    parser.add_argument("--pdf-crop", action="store_true", help="Crop PDF by TOC (intro~conclusion); default: off, use full PDF")
    parser.add_argument("--multimodal", default="false", choices=["true", "false"], help="Use multimodal classification prompt (default: false)")

    args = parser.parse_args()
    default_model = args.model
    model_config = {
        "default": default_model,
        "sourcing": args.model_sourcing or default_model,
        "tracing": args.model_tracing or default_model,
        "paper": args.model_paper or default_model,
        "classification": args.model_classification or default_model,
        "aggregation": args.model_aggregation or default_model,
        "dataset_builder": args.model_dataset_builder or default_model,
    }
    max_depth = None if args.max_depth == -1 else args.max_depth
    enable_blog = args.blog_analysis and not args.no_blog_analysis
    enable_paper = args.paper_analysis and not args.no_paper_analysis
    enable_pdf_crop = args.pdf_crop
    enable_multimodal = args.multimodal.lower() == "true"

    try:
        analyze_datasets(
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
            max_depth=max_depth,
            model_config=model_config,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
            log_level=args.log_level,
            load_existing=not args.no_load_existing,
            api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            enable_blog_analysis=enable_blog,
            enable_paper_analysis=enable_paper,
            enable_multimodal=enable_multimodal,
            enable_pdf_crop=enable_pdf_crop
        )
            
    except KeyboardInterrupt:
        print("\n\nUser interrupted execution")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nExecution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
