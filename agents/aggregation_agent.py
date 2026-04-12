"""
Aggregation Agent - aggregates and integrates dataset analysis results.

Includes DatasetValidator, DatasetFilter, DatasetDeduplicator as supporting components.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher

from ..models import DatasetExistence
from ..utils import (
    safe_requests_get,
    extract_year_month_from_paper_links,
    summarize_source_relationships,
    invoke_json_llm,
    parse_llm_json_response,
)

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Dataset validator - handles existence verification and alternative version search"""

    DATASET_ALIASES = {
        'math': 'EleutherAI/hendrycks_math',
        'mathematics': 'EleutherAI/hendrycks_math',
        'math_dataset': 'EleutherAI/hendrycks_math',
        'hendrycks_math': 'EleutherAI/hendrycks_math',
        'competition_math': 'qwedsacf/competition_math',
        'competition': 'qwedsacf/competition_math',
        'evol_instruct': 'WizardLMTeam/WizardLM_evol_instruct_70k',
        'evol': 'WizardLMTeam/WizardLM_evol_instruct_70k',
        'LOGIC-701': "hivaze/LOGIC-701",
        'Chinese K-12 Exam': 'aslawliet/cn-k12',
        'K-12 Exam': 'aslawliet/cn-k12',
        'K-12': 'aslawliet/cn-k12',
    }

    DATASET_BLACKLIST = set()

    def __init__(self, api_client, llm=None):
        self.api_client = api_client
        self.llm = llm
        self._existence_cache = {}
        self._search_cache = {}
        self._mapping_cache = {}
        self._time_cache = {}

    @staticmethod
    def _strip_citation_suffix(dataset_name: str) -> str:
        """Remove trailing citation-style suffixes such as '(2021a)' or '(2017)' from dataset names."""
        if not dataset_name or '(' not in dataset_name:
            return dataset_name.strip()
        # Match a trailing (YYYY) or (YYYYx) suffix, such as (2017), (2021a), or (2022b).
        stripped = re.sub(r'\s*\(\d{4}[a-z]?\)\s*$', '', dataset_name, flags=re.IGNORECASE).strip()
        return stripped if stripped else dataset_name.strip()

    def _is_in_blacklist(self, dataset_name: str) -> bool:
        if '/' in dataset_name:
            core_name = dataset_name.split('/', 1)[1]
        else:
            core_name = dataset_name
        normalized_name = self._normalize_for_lookup(core_name)
        for blacklist_item in self.DATASET_BLACKLIST:
            if self._normalize_for_lookup(blacklist_item) == normalized_name:
                return True
        return False

    def verify_existence(self, dataset_name: str) -> DatasetExistence:
        if dataset_name in self._existence_cache:
            return self._existence_cache[dataset_name]
        if self._is_in_blacklist(dataset_name):
            self._existence_cache[dataset_name] = DatasetExistence.NOT_FOUND
            return DatasetExistence.NOT_FOUND
        try:
            if '/' not in dataset_name:
                self._existence_cache[dataset_name] = DatasetExistence.NOT_FOUND
                return DatasetExistence.NOT_FOUND
            api_info = self.api_client._get_dataset_info_via_api(dataset_name)
            if api_info:
                self._existence_cache[dataset_name] = DatasetExistence.EXISTS
                return DatasetExistence.EXISTS
            response = safe_requests_get(
                f"https://huggingface.co/datasets/{dataset_name}",
                timeout=10, max_retries=3, sleep_time=10
            )
            if response.status_code == 200:
                self._existence_cache[dataset_name] = DatasetExistence.EXISTS
                return DatasetExistence.EXISTS
            self._existence_cache[dataset_name] = DatasetExistence.NOT_FOUND
            return DatasetExistence.NOT_FOUND
        except Exception:
            self._existence_cache[dataset_name] = DatasetExistence.UNCERTAIN
            return DatasetExistence.UNCERTAIN

    def resolve_missing_dataset(
        self,
        dataset_name: str,
        evidence: Optional[str] = None,
        source_type: Optional[str] = None,
        parent_dataset: Optional[str] = None,
        **extra_context: Any,
    ) -> Optional[str]:
        """Resolve a name not on HF: alias -> strip suffix + search -> LLM infer candidates then search. Returns canonical HF name if found."""
        logger.info(
            f"[Resolve] Start resolving '{dataset_name}' "
            f"(source={source_type or 'unknown'}, parent={parent_dataset or 'unknown'})"
        )
        if self._is_in_blacklist(dataset_name):
            logger.info(f"[Resolve] '{dataset_name}' is blacklisted")
            return None
        mapped = self._lookup_dataset_alias(dataset_name)
        if mapped and self.verify_existence(mapped) == DatasetExistence.EXISTS:
            logger.info(f"[Resolve] Alias matched '{dataset_name}' -> '{mapped}'")
            return mapped
        core_name = self._strip_citation_suffix(dataset_name)
        search_name = core_name if core_name else dataset_name
        direct_results = self._search_huggingface_direct(search_name)
        if direct_results:
            best = self._select_best_match(
                search_name,
                direct_results,
                evidence=evidence,
                source_type=source_type,
                parent_dataset=parent_dataset,
                **extra_context,
            )
            if best and self.verify_existence(best) == DatasetExistence.EXISTS:
                logger.info(f"[Resolve] Direct search matched '{dataset_name}' -> '{best}'")
                return best
        for candidate in self._infer_candidate_names_with_llm(
            search_name,
            evidence=evidence,
            source_type=source_type,
            parent_dataset=parent_dataset,
            **extra_context,
        ):
            logger.info(f"[Resolve] LLM candidate for '{dataset_name}': {candidate}")
            search_results = self._search_huggingface_direct(candidate)
            if search_results:
                best = self._select_best_match(
                    candidate,
                    search_results,
                    evidence=evidence,
                    source_type=source_type,
                    parent_dataset=parent_dataset,
                    original_dataset_name=dataset_name,
                    **extra_context,
                )
                if best and self.verify_existence(best) == DatasetExistence.EXISTS:
                    logger.info(f"[Resolve] LLM-assisted match '{dataset_name}' -> '{best}'")
                    return best
        logger.info(f"[Resolve] Failed to resolve '{dataset_name}'")
        return None

    def _normalize_for_lookup(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[\s_\-\.]+', '', text)
        return re.sub(r'[^a-z0-9]', '', text)

    def _lookup_dataset_alias(self, dataset_name: str) -> Optional[str]:
        if dataset_name in self._mapping_cache:
            return self._mapping_cache[dataset_name]
        norm = self._normalize_for_lookup(dataset_name)
        for alias, standard in self.DATASET_ALIASES.items():
            if self._normalize_for_lookup(alias) == norm:
                self._mapping_cache[dataset_name] = standard
                return standard
        return None

    def _search_huggingface_direct(self, dataset_name: str) -> List[str]:
        if dataset_name in self._search_cache:
            return self._search_cache[dataset_name]
        try:
            query = self._extract_search_keywords(dataset_name)
            resp = safe_requests_get(
                "https://huggingface.co/api/datasets",
                params={'search': query, 'limit': 20, 'sort': 'likes'},
                timeout=15, max_retries=5, sleep_time=30
            )
            if resp.status_code == 200:
                results = [i.get('id', '') for i in resp.json() if i.get('id')]
                self._search_cache[dataset_name] = [r for r in results if r]
                logger.info(
                    f"[Resolve] HF search '{dataset_name}' (query='{query}') -> "
                    f"{self._search_cache[dataset_name][:8]}"
                )
                return self._search_cache[dataset_name]
        except Exception:
            pass
        self._search_cache[dataset_name] = []
        logger.info(f"[Resolve] HF search '{dataset_name}' returned no candidates")
        return []

    def _extract_search_keywords(self, dataset_name: str) -> str:
        core = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
        core = re.sub(r'[-_](v?\d+|data|dataset|corpus)$', '', core, flags=re.IGNORECASE)
        return re.sub(r'^(dataset|data)[-_]', '', core, flags=re.IGNORECASE)

    def _select_best_match(
        self,
        original_name: str,
        candidates: List[str],
        evidence: Optional[str] = None,
        source_type: Optional[str] = None,
        parent_dataset: Optional[str] = None,
        **extra_context: Any,
    ) -> Optional[str]:
        if not candidates:
            return None

        def norm(s):
            return re.sub(r'[\s_\-]', '', s).lower()

        def ds_part(name):
            return name.split('/', 1)[1] if '/' in name else name

        def org_part(name):
            return name.split('/', 1)[0] if '/' in name else ""

        orig_ds = ds_part(original_name)
        n_orig = norm(original_name)
        n_orig_ds = norm(orig_ds)

        for c in candidates:
            if norm(c) == n_orig:
                logger.info(f"[Resolve] Heuristic exact match '{original_name}' -> '{c}'")
                return c

        exact_ds_matches = [c for c in candidates if norm(ds_part(c)) == n_orig_ds]
        if exact_ds_matches:
            logger.info(
                f"[Resolve] Heuristic dataset-name exact matches for '{original_name}' -> "
                f"{exact_ds_matches[:5]} (choose first by search ranking)"
            )
            return exact_ds_matches[0]

        for c in candidates:
            nd = norm(ds_part(c))
            if n_orig_ds in nd or nd in n_orig_ds:
                logger.info(f"[Resolve] Heuristic fuzzy match '{original_name}' -> '{c}'")
                return c
        org_matches = []
        for c in candidates:
            co = org_part(c)
            cd = ds_part(c)
            if n_orig_ds in norm(co) and not (n_orig_ds in norm(cd) or norm(cd) in n_orig_ds):
                ratio = SequenceMatcher(None, orig_ds.lower(), co.lower()).ratio()
                org_matches.append((c, ratio))
        if org_matches:
            org_matches.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"[Resolve] Heuristic org-based match '{original_name}' -> '{org_matches[0][0]}'")
            return org_matches[0][0]

        llm_choice = self._select_best_match_with_llm(
            original_name,
            candidates,
            evidence=evidence,
            source_type=source_type,
            parent_dataset=parent_dataset,
            **extra_context,
        )
        if llm_choice:
            return llm_choice
        return None

    def _select_best_match_with_llm(
        self,
        original_name: str,
        candidates: List[str],
        evidence: Optional[str] = None,
        source_type: Optional[str] = None,
        parent_dataset: Optional[str] = None,
        **extra_context: Any,
    ) -> Optional[str]:
        if not self.llm or len(candidates) <= 1:
            return candidates[0] if len(candidates) == 1 else None

        try:
            from ..prompts import PROMPTS

            prompt_tpl = PROMPTS['dataset_matching']
            parts = []
            if evidence and evidence.strip():
                parts.append(f"Evidence: {evidence.strip()}")
            if source_type and source_type.strip():
                parts.append(f"Source type: {source_type.strip()}")
            if parent_dataset and parent_dataset.strip():
                parts.append(f"Parent dataset: {parent_dataset.strip()}")
            for key, value in extra_context.items():
                if value is not None and str(value).strip():
                    parts.append(f"{key}: {str(value).strip()}")
            context_block = "\n".join(parts) if parts else "No additional context."

            response = invoke_json_llm(
                self.llm,
                prompt_tpl.format(
                    original_name=original_name,
                    context_block=context_block,
                    candidates=json.dumps(candidates, ensure_ascii=False),
                ),
                logger=logger,
                context=f"Dataset matching for {original_name}",
            )
            selected = None
            parsed = parse_llm_json_response(
                response.content,
                logger=logger,
                context=f"Dataset matching for {original_name}",
            )
            selected = parsed.get('selected_name')
            if selected is None:
                response_text = response.content.strip()
                if '```json' in response_text:
                    response_text = response_text.split('```json', 1)[1].split('```', 1)[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```', 1)[1].split('```', 1)[0].strip()
                selected = response_text.strip().strip('"').strip("'")

            if selected is None:
                return None
            selected = str(selected).strip().strip('"').strip("'")
            if selected.lower() == "null":
                logger.info(f"[Resolve] LLM matcher rejected all candidates for '{original_name}'")
                return None

            for candidate in candidates:
                if candidate == selected:
                    logger.info(f"[Resolve] LLM matcher selected '{candidate}' for '{original_name}'")
                    return candidate

            normalized = re.sub(r'[\s_\-]', '', selected).lower()
            for candidate in candidates:
                if re.sub(r'[\s_\-]', '', candidate).lower() == normalized:
                    logger.info(f"[Resolve] LLM matcher normalized '{selected}' -> '{candidate}'")
                    return candidate
        except Exception:
            return None

        return None

    def _infer_candidate_names_with_llm(
        self,
        dataset_name: str,
        evidence: Optional[str] = None,
        source_type: Optional[str] = None,
        parent_dataset: Optional[str] = None,
        **extra_context: Any,
    ) -> List[str]:
        """Infer 3–5 candidate HuggingFace dataset names from the given name and optional context."""
        if not self.llm:
            return []
        try:
            from ..prompts import PROMPTS
            prompt_tpl = PROMPTS['dataset_name_inference']
            parts = []
            if evidence and evidence.strip():
                parts.append(f"**Evidence (quote where this dataset is mentioned):**\n{evidence.strip()}")
            if source_type and source_type.strip():
                parts.append(f"**Source:** {source_type.strip()}")
            if parent_dataset and parent_dataset.strip():
                parts.append(f"**Parent dataset (where this source was cited):** {parent_dataset.strip()}")
            for k, v in extra_context.items():
                if v is not None and str(v).strip():
                    parts.append(f"**{k}:** {str(v).strip()}")
            context_block = "\n".join(parts) if parts else "No additional context."
            prompt = prompt_tpl.format(dataset_name=dataset_name, context_block=context_block)
            resp = self.llm.invoke(prompt)
            candidates = []
            for line in resp.content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                if '/' in line:
                    part = line.split('/')[-1]
                    if len(part) > 2:
                        candidates.append(part)
                elif len(line) > 2:
                    candidates.append(line)
            return candidates[:5]
        except Exception:
            return []


class DatasetFilter:
    """Filters out results that are clearly not datasets (e.g., model names)."""

    def filter_non_datasets(self, relationships: List[Dict]) -> List[Dict]:
        if not relationships:
            return relationships
        return [r for r in relationships if self._is_likely_dataset(r['name'])]

    def _is_likely_dataset(self, name: str) -> bool:
        if '/' in name:
            name = name.split('/', 1)[1]
        for pat in [r'-\d+[bB]$', r'-(chat|base|2507)$']:
            if re.search(pat, name, re.IGNORECASE):
                return False
        return True


class DatasetDeduplicator:
    """Merges similar datasets."""

    def __init__(self):
        self.similarity_threshold = 0.9

    @staticmethod
    def _confidence_value(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def deduplicate_datasets(self, relationships: List[Dict]) -> List[Dict]:
        if len(relationships) <= 1:
            return relationships
        groups = []
        processed = set()
        for i, relationship in enumerate(relationships):
            if i in processed:
                continue
            group = [relationship]
            processed.add(i)
            for j, other in enumerate(relationships[i + 1:], i + 1):
                if j in processed:
                    continue
                if self._are_similar(relationship['name'], other['name']):
                    group.append(other)
                    processed.add(j)
            groups.append(group)
        return [self._merge_group(group) for group in groups]

    def _merge_group(self, relationships: List[Dict]) -> Dict:
        if len(relationships) == 1:
            return relationships[0]

        best_name = self._select_best([r['name'] for r in relationships])
        best_relationship = max(
            relationships,
            key=lambda item: (
                1 if item.get('name') == best_name else 0,
                1 if item.get('exists_on_hf') == DatasetExistence.EXISTS else 0,
                self._confidence_value(item.get('confidence', 0)),
            )
        )

        merged = best_relationship.copy()
        merged['name'] = best_name
        merged['confidence'] = max(
            (self._confidence_value(item.get('confidence', 0)) for item in relationships),
            default=0.0,
        )

        evidences = []
        seen_evidence = set()
        for item in relationships:
            evidence = (item.get('evidence') or '').strip()
            if evidence and evidence not in seen_evidence:
                seen_evidence.add(evidence)
                evidences.append(evidence)
        if evidences:
            merged['evidence'] = " || ".join(evidences)

        sources = []
        seen_sources = set()
        for item in relationships:
            source = (item.get('source') or '').strip()
            if source and source not in seen_sources:
                seen_sources.add(source)
                sources.append(source)
        if sources:
            merged['source'] = "; ".join(sources)

        existence_values = [item.get('exists_on_hf') for item in relationships]
        if DatasetExistence.EXISTS in existence_values:
            merged['exists_on_hf'] = DatasetExistence.EXISTS
        elif DatasetExistence.UNCERTAIN in existence_values:
            merged['exists_on_hf'] = DatasetExistence.UNCERTAIN
        elif existence_values:
            merged['exists_on_hf'] = DatasetExistence.NOT_FOUND

        original_names = []
        seen_originals = set()
        for item in relationships:
            original_name = (item.get('original_name') or '').strip()
            if original_name and original_name not in seen_originals:
                seen_originals.add(original_name)
                original_names.append(original_name)
        if original_names:
            merged['original_name'] = original_names[0]
            merged['merged_aliases'] = original_names

        raw_names = []
        seen_raw_names = set()
        for item in relationships:
            raw_name = (item.get('name') or '').strip()
            if raw_name and raw_name not in seen_raw_names:
                seen_raw_names.add(raw_name)
                raw_names.append(raw_name)
        if len(raw_names) > 1:
            merged['merged_names'] = raw_names

        return merged

    def _group_similar(self, datasets: List[str]) -> List[List[str]]:
        groups = []
        processed = set()
        for i, d in enumerate(datasets):
            if i in processed:
                continue
            group = [d]
            processed.add(i)
            for j, o in enumerate(datasets[i + 1:], i + 1):
                if j in processed:
                    continue
                if self._are_similar(d, o):
                    group.append(o)
                    processed.add(j)
            groups.append(group)
        return groups

    def _are_similar(self, a: str, b: str) -> bool:
        nums_a = re.findall(r'\b\d+\b', a)
        nums_b = re.findall(r'\b\d+\b', b)
        if nums_a and nums_b:
            na = [n for n in nums_a if len(n) != 4]
            nb = [n for n in nums_b if len(n) != 4]
            if na and nb and set(na) != set(nb):
                return False
        for pat in [r'[vV](\d+(?:\.\d+)?)', r'version[-_]?(\d+(?:\.\d+)?)']:
            m1, m2 = re.search(pat, a), re.search(pat, b)
            if m1 and m2 and m1.group(1) != m2.group(1):
                return False
        if '/' in a and '/' in b and a.split('/', 1)[0] != b.split('/', 1)[0]:
            return False
        y1 = set(re.findall(r'\b(19\d{2}|20\d{2})\b', a))
        y2 = set(re.findall(r'\b(19\d{2}|20\d{2})\b', b))
        if y1 and y2 and y1 != y2:
            return False
        return SequenceMatcher(None, a, b).ratio() >= self.similarity_threshold

    def _select_best(self, group: List[str]) -> str:
        if len(group) == 1:
            return group[0]
        return max(group, key=lambda n: (1 if '/' in n else 0) - len(n) / 100)


class AggregationAgent:
    """
    Aggregation Agent - aggregates and integrates dataset analysis results.

    Pipeline: existence verification -> deduplication -> resolve missing ->
    filter model names -> temporal order validation -> re-deduplication.
    """

    def __init__(self, validator: DatasetValidator, deduplicator: DatasetDeduplicator):
        self.validator = validator
        self.deduplicator = deduplicator
        self.filter = DatasetFilter()
        self._time_cache = {}

    def integrate_results(
        self,
        target_dataset: str,
        candidates: List[Dict],
        target_year: Optional[str] = None,
    ) -> List[Dict]:
        logger.info(f"Starting integration: {target_dataset} ({len(candidates)} candidates)")
        logger.info(
            f"Integration input for {target_dataset}: "
            f"{summarize_source_relationships(candidates, limit=12)}"
        )
        for c in candidates:
            c['exists_on_hf'] = self.validator.verify_existence(c['name'])
        result = self.deduplicator.deduplicate_datasets(candidates)
        resolved = []
        for c in result:
            if c['exists_on_hf'] == DatasetExistence.EXISTS:
                resolved.append(c)
            else:
                orig = c['name']
                r = self.validator.resolve_missing_dataset(
                    orig,
                    evidence=c.get('evidence'),
                    source_type=c.get('source'),
                    parent_dataset=target_dataset,
                )
                if r:
                    c['original_name'] = orig
                    c['name'] = r
                    c['exists_on_hf'] = DatasetExistence.EXISTS
                resolved.append(c)
        resolved = self.filter.filter_non_datasets(resolved)
        logger.info(
            f"Post-resolution candidates for {target_dataset}: "
            f"{summarize_source_relationships(resolved, limit=12)}"
        )
        if target_year:
            resolved = self._validate_temporal_order(target_dataset, target_year, resolved)
            logger.info(
                f"Post-temporal-validation candidates for {target_dataset}: "
                f"{summarize_source_relationships(resolved, limit=12)}"
            )
        return self.deduplicator.deduplicate_datasets(resolved)

    def _validate_temporal_order(
        self, target_dataset: str, target_year: str, relationships: List[Dict]
    ) -> List[Dict]:
        try:
            ym = target_year if '-' in target_year else f"{target_year}-12"
            parts = ym.split('-')
            if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
                return relationships
        except (ValueError, AttributeError, TypeError):
            return relationships
        out = []
        for r in relationships:
            if r.get('exists_on_hf') != DatasetExistence.EXISTS:
                out.append(r)
                continue

            source_time_info = self._get_source_time_info(r['name'])
            hf_year = source_time_info.get('hf_year')
            paper_year = source_time_info.get('paper_year')
            src_ym = source_time_info.get('effective_year')

            if not hf_year or not paper_year or not src_ym:
                logger.debug(
                    f"[Temporal] Skip filtering for {r['name']} because source times are incomplete: "
                    f"hf_year={hf_year}, paper_year={paper_year}"
                )
                out.append(r)
                continue

            if src_ym > ym:
                if r.get('original_name'):
                    logger.info(
                        f"[Temporal] Reverting mapping {r.get('original_name')} -> {r['name']} "
                        f"because source_time={src_ym} > target_time={ym}"
                    )
                    r['name'] = r['original_name']
                    r['exists_on_hf'] = DatasetExistence.NOT_FOUND
                    r.pop('original_name', None)
                    out.append(r)
            else:
                r.pop('original_name', None)
                out.append(r)
        return out

    def _get_source_time_info(self, source_dataset: str) -> Dict[str, Optional[str]]:
        if source_dataset in self._time_cache:
            return self._time_cache[source_dataset]

        result = {
            "hf_year": None,
            "paper_year": None,
            "effective_year": None,
        }
        try:
            info = self.validator.api_client._get_dataset_info_via_api(source_dataset)
            if info:
                result["hf_year"] = self.validator.api_client._extract_year_from_api_info(info)

            readme = self.validator.api_client._get_readme_via_api(source_dataset)
            if readme:
                paper_links = self.validator.api_client._extract_paper_links(readme)
                result["paper_year"] = extract_year_month_from_paper_links(paper_links)

            if result["hf_year"] and result["paper_year"]:
                result["effective_year"] = min(result["hf_year"], result["paper_year"])
        except Exception:
            pass

        self._time_cache[source_dataset] = result
        return result
