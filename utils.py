import time
import json
import logging
import re
import requests
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}


def extract_year_month_from_paper_url(paper_url: str) -> Optional[str]:
    """Extract approximate publication year-month from common paper URLs."""
    if not paper_url:
        return None

    url = paper_url.strip()
    if not url:
        return None

    try:
        arxiv_match = re.search(
            r'(?:arxiv\.org/(?:abs|pdf)/|huggingface\.co/papers/)(\d{2})(\d{2})\.\d+',
            url,
            re.IGNORECASE,
        )
        if arxiv_match:
            yy, month = int(arxiv_match.group(1)), int(arxiv_match.group(2))
            if 1 <= month <= 12:
                year = 1900 + yy if yy >= 91 else 2000 + yy
                return f"{year}-{month:02d}"

        acl_match = re.search(r'aclanthology\.org/(\d{4})\.[^/\s]+', url, re.IGNORECASE)
        if acl_match:
            return f"{int(acl_match.group(1)):04d}-01"

        old_acl_match = re.search(r'aclanthology\.org/[A-Z]?(\d{2})-\d+', url, re.IGNORECASE)
        if old_acl_match:
            yy = int(old_acl_match.group(1))
            year = 1900 + yy if yy >= 91 else 2000 + yy
            return f"{year:04d}-01"
    except Exception:
        return None

    return None


def extract_year_month_from_paper_links(paper_links: List[str]) -> Optional[str]:
    """Extract the earliest available publication year-month from a list of paper links."""
    years = []
    for url in paper_links or []:
        ym = extract_year_month_from_paper_url(url)
        if ym:
            years.append(ym)

    if not years:
        return None

    years.sort()
    return years[0]


def make_text_preview(text: Optional[str], max_len: int = 400) -> str:
    """Return a compact one-line preview for logging."""
    value = re.sub(r"\s+", " ", (text or "").strip())
    if not value:
        return ""
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + " ..."


def summarize_source_relationships(relationships: List[dict], limit: int = 8) -> str:
    """Build a compact summary string for source relationship logs."""
    if not relationships:
        return "none"

    items = []
    for rel in relationships[:limit]:
        name = (rel.get("name") or "").strip() or "<unknown>"
        confidence = rel.get("confidence")
        if isinstance(confidence, (int, float)):
            items.append(f"{name}({confidence:.2f})")
        else:
            items.append(name)

    if len(relationships) > limit:
        items.append(f"... +{len(relationships) - limit} more")
    return ", ".join(items)


def invoke_json_llm(llm: Any, messages: Any, logger: Optional[logging.Logger] = None, context: str = "") -> Any:
    """Invoke an LLM with JSON mode when available, falling back to normal invoke."""
    try:
        return llm.bind(response_format={"type": "json_object"}).invoke(messages)
    except Exception as exc:
        if logger:
            prefix = f"{context}: " if context else ""
            logger.debug(f"{prefix}JSON-mode invoke unavailable, falling back to normal invoke: {exc}")
        return llm.invoke(messages)


def _strip_code_fences(text: str) -> str:
    value = (text or "").strip()
    if "```json" in value:
        return value.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in value:
        return value.split("```", 1)[1].split("```", 1)[0].strip()
    return value


def _normalize_json_text(text: str) -> str:
    value = (text or "").strip()
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value.strip()


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_balanced_json_substring(text: str) -> Optional[str]:
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if start is None and char not in "{[":
            continue

        if start is None:
            start = idx
            depth = 1
            in_string = False
            escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    if start is not None:
        return text[start:]
    return None


def _salvage_string_array_field(text: str, field_name: str) -> List[str]:
    pattern = re.compile(
        rf'"{re.escape(field_name)}"\s*:\s*\[(.*?)\]',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    raw_block = match.group(1)
    values = re.findall(r'"([^"\n]+)"', raw_block)
    seen = set()
    results = []
    for value in values:
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            results.append(item)
    return results


def _salvage_source_datasets(text: str) -> List[Dict[str, Any]]:
    entries = []
    seen = set()
    pattern = re.compile(
        r'"name"\s*:\s*"([^"\n]+)"'
        r'(?:(?:(?!\{|\}).)*)?"relationship"\s*:\s*"([^"\n]+)"'
        r'(?:(?:(?!\{|\}).)*)?"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
        r'(?:(?:(?!\{|\}).)*)?"evidence"\s*:\s*"([^"\n]*)',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        name = match.group(1).strip()
        relationship = match.group(2).strip()
        confidence = match.group(3).strip()
        evidence = match.group(4).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            confidence_value = float(confidence)
        except ValueError:
            confidence_value = 0.0
        entries.append(
            {
                "name": name,
                "relationship": relationship or "Direct Inclusion/Subset",
                "confidence": confidence_value,
                "evidence": evidence,
            }
        )
    return entries


def parse_llm_json_response(
    response_text: str,
    logger: Optional[logging.Logger] = None,
    context: str = "LLM response",
) -> Dict[str, Any]:
    """Best-effort parser for LLM JSON responses with light repair and salvage."""
    raw_text = response_text or ""
    candidates = []

    for candidate in (
        raw_text,
        _strip_code_fences(raw_text),
        _extract_balanced_json_substring(_strip_code_fences(raw_text) or raw_text),
    ):
        if not candidate:
            continue
        normalized = _normalize_json_text(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    for candidate in candidates:
        for variant in (candidate, _remove_trailing_commas(candidate)):
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"items": parsed}
            except json.JSONDecodeError:
                continue

    salvaged: Dict[str, Any] = {}
    source_datasets = _salvage_source_datasets(raw_text)
    if source_datasets:
        salvaged["source_datasets"] = source_datasets

    for field_name in ("paper_links", "github_links", "blog_links", "categories"):
        values = _salvage_string_array_field(raw_text, field_name)
        if values:
            salvaged[field_name] = values

    selected_match = re.search(r'"selected_name"\s*:\s*"([^"\n]+)"', raw_text, re.IGNORECASE)
    if selected_match:
        salvaged["selected_name"] = selected_match.group(1).strip()

    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_text, re.IGNORECASE)
    if confidence_match:
        try:
            salvaged["confidence"] = float(confidence_match.group(1))
        except ValueError:
            pass

    if salvaged:
        if logger:
            logger.warning(
                f"{context} JSON parsing failed; salvaged partial result. "
                f"Response preview: {make_text_preview(raw_text, max_len=300)}"
            )
        return salvaged

    if logger:
        logger.warning(
            f"{context} JSON parsing failed with no salvage. "
            f"Response preview: {make_text_preview(raw_text, max_len=300)}"
        )
    return {}

def safe_requests_get(
    url: str, 
    params: Optional[dict] = None,
    timeout: int = 15,
    max_retries: int = 5,
    sleep_time: int = 30,
    **kwargs
) -> requests.Response:
    """
    requests.get with retry mechanism
    
    Args:
        url: Request URL
        params: Request parameters
        timeout: Timeout in seconds
        max_retries: Maximum retry attempts
        sleep_time: Base wait time between retries
        **kwargs: Other requests parameters
    
    Returns:
        requests.Response object
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout, **kwargs)
            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt < max_retries:
                    wait_time = sleep_time * (2 ** attempt)
                    logger.warning(
                        f"HTTP GET retryable status {response.status_code} "
                        f"(attempt {attempt + 1}/{max_retries + 1}) for {url}"
                    )
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"HTTP GET failed with retryable status {response.status_code} after retries: {url}")
            return response
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.SSLError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            last_exception = e
            
            error_str = str(e)
            if any(keyword in error_str for keyword in [
                'HTTPSConnectionPool', 
                'SSLError', 
                'ConnectionError',
                'Read timed out',
                'Max retries exceeded'
            ]):
                if attempt < max_retries:
                    wait_time = sleep_time * (2 ** attempt)
                    logger.warning(f"Network request failed (attempt {attempt + 1}/{max_retries + 1}): {error_str}")
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Network request finally failed after {max_retries} retries: {error_str}")
                    raise e
            else:
                raise e
        except Exception as e:
            raise e
    
    if last_exception:
        raise last_exception

def safe_requests_post(
    url: str, 
    data: Optional[Any] = None,
    json: Optional[dict] = None,
    timeout: int = 15,
    max_retries: int = 5,
    sleep_time: int = 30,
    **kwargs
) -> requests.Response:
    """
    requests.post with retry mechanism
    
    Args:
        url: Request URL
        data: Request data
        json: JSON data
        timeout: Timeout in seconds
        max_retries: Maximum retry attempts
        sleep_time: Base wait time between retries
        **kwargs: Other requests parameters
    
    Returns:
        requests.Response object
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, data=data, json=json, timeout=timeout, **kwargs)
            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt < max_retries:
                    wait_time = sleep_time * (2 ** attempt)
                    logger.warning(
                        f"HTTP POST retryable status {response.status_code} "
                        f"(attempt {attempt + 1}/{max_retries + 1}) for {url}"
                    )
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"HTTP POST failed with retryable status {response.status_code} after retries: {url}")
            return response
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.SSLError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            last_exception = e
            
            error_str = str(e)
            if any(keyword in error_str for keyword in [
                'HTTPSConnectionPool', 
                'SSLError', 
                'ConnectionError',
                'Read timed out',
                'Max retries exceeded'
            ]):
                if attempt < max_retries:
                    wait_time = sleep_time * (2 ** attempt)
                    logger.warning(f"Network request failed (attempt {attempt + 1}/{max_retries + 1}): {error_str}")
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Network request finally failed after {max_retries} retries: {error_str}")
                    raise e
            else:
                raise e
        except Exception as e:
            raise e
    
    if last_exception:
        raise last_exception
