"""
Data lineage analysis prompt templates.
Each prompt is designed for a specific stage of dataset lineage analysis.
"""

# ---------------------------------------------------------------------------
# Shared constants for tracing prompts
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPES = """
**relationship** MUST be exactly one of:
1. **Semantic Evolution** — Reformulating or enhancing the original question while allowing controlled semantic variation
2. **CoT Distillation** — Keeping the question unchanged while using a stronger teacher model to generate long CoT responses
3. **Synthetic Generation** — Using upstream data as seeds to generate new question-answer pairs through LLM generalization
4. **Structured Fusion** — Concatenating or combining data from multiple sources for composite reasoning
5. **Direct Inclusion/Subset** — Including upstream data as-is as part of the new dataset
"""

_CORE = """
## Core Principle
A dataset is a **source** ONLY IF the text explicitly states that data was
taken from, derived from, sampled from, collected from, extended using,
or merged/combined from it, OR it is listed as a **subset / component / sub-dataset**
of the target (e.g., a "Subsets" or "Components" section listing named parts).

This is **data lineage** — not design influence, not citation, not evaluation.

A dataset is **NOT** a source if the text merely:
- follows its design principles or annotation rules,
- uses a similar pipeline or methodology,
- compares performance to it,
- cites it as related work,
- contrasts with it, or is inspired by it.

No inference or assumption beyond explicit statements is allowed.
Do NOT infer external datasets unless the text explicitly states data was taken from them.
"""

_INCLUDE = """
## Include as Source ONLY IF:
1. Data was directly taken, derived, sampled, collected, or filtered from it.
2. It was merged or combined to form the target.
3. It is explicitly described as base material for construction.
4. It appears in an explicit "data sources" / "source datasets" / "subsets" / "components" / "sub-datasets" list.

When a "Subsets" or similar section lists named parts of the target
(e.g., "Prefill (43K)", "Leetcode (27K)"), treat each part as a source dataset.
"""

_EXCLUDE = """
## Exclude:
- Evaluation / benchmarking datasets (even if listed in "Evaluation" sections)
- Comparison baselines, related work, citations
- Design inspiration or methodology references
- Any dataset with no explicit data-usage statement or component listing
- The target dataset itself
- Models, organizations, or institutions
- **Generic category terms** (NOT actual dataset names):
  e.g., "Synthetic Data", "Training Data", "Source Data", "Raw Data",
  "External Data", "Public Data", "Collected Data", "Base Data".
  When text uses such a heading followed by specific names
  (e.g., "Synthetic Data: We use MATH and AMC-AIME"),
  extract the specific names ("MATH", "AMC-AIME"), NOT the heading.
"""

_VERIFY = """
## Verification (ALL must pass):
1. **Presence** — Name appears in the text.
2. **Explicit Usage** — Text states data was used/derived/merged, or lists it
   as a source/subset/component. (A "Subsets" listing counts.)
3. **Data Origin** — Contributed actual data samples, not just design ideas,
   pipelines, or principles.
4. **Specificity** — Must be a specific dataset name, not a generic term.
   Valid: "MATH", "AMC-AIME", "Common Crawl", "Wikipedia".
   Invalid: "Synthetic Data", "Training Data", "Source Data".
   If text says "Synthetic Data: We use MATH and AMC-AIME",
   extract "MATH" and "AMC-AIME", NOT "Synthetic Data".
5. **Non-Design/Non-Eval** — Not mentioned only in contexts like
   "similar to", "inspired by", "follows the same principles",
   "pipeline borrowed from", comparison, evaluation, or citation.

Any check fails → exclude.

## Extraction Examples:
CORRECT:
- "We use MATH dataset and AMC-AIME dataset to create synthetic problems"
  → ["MATH", "AMC-AIME"]
- "Synthetic Data: We follow Li et al. to create synthetic math data using the MATH dataset"
  → ["MATH"] (NOT "Synthetic Data")

INCORRECT (DO NOT DO THIS):
- "Synthetic Data: We use MATH and AMC-AIME" → WRONG: ["Synthetic Data"]
  CORRECT: ["MATH", "AMC-AIME"]
- "Training data comes from multiple sources" → WRONG: ["Training data"]
  CORRECT: [] (no specific dataset names)
"""

_NAME_EXTRACT = """
## Name Extraction:
- **When the text contains an explicit HuggingFace dataset link** (e.g. https://huggingface.co/datasets/org/dataset-name or [org/dataset-name](url)), you MUST use the canonical form **org/dataset-name** exactly as given by that link (the last path segment pair: organization and dataset name). Do not alter or guess a different form.
- If no HuggingFace link exists, return the name exactly as written in the text.
- Do NOT guess organization names or expand abbreviations when no link is provided.
"""

_OUTPUT = """
## Confidence Scoring:
- **0.9–1.0** — Explicit data usage statement ("we use/include/derive from [dataset]"), OR listed in a data sources / subsets / components section.
- **0.7–0.9** — Mentioned in construction context with clear but indirect evidence (e.g., appears in a data pipeline description, co-listed with confirmed sources).
- **0.5–0.7** — Context suggests usage but is ambiguous (could be evaluation, could be training).
- **< 0.5** — Do NOT include (verification would likely fail).

## Output
Return ONLY valid JSON:
{{
    "source_datasets": [
        {{
            "name": "source-dataset-name",
            "relationship": "Semantic Evolution | CoT Distillation | Synthetic Generation | Structured Fusion | Direct Inclusion/Subset",
            "confidence": 0.0-1.0,
            "evidence": "Verbatim quote showing explicit construction/source usage"
        }}
    ],
    "reasoning": "How source datasets were identified and verified"
}}

If none found: {{"source_datasets": []}}
- `evidence` MUST come from a construction/source paragraph, source-data list intro, or explicit source table context.
- NEVER use evidence from license/legal/compliance/copyright/acknowledgement text.
- NEVER use appendix-only metadata or per-dataset license rows as evidence unless the surrounding text explicitly says these listed datasets are source datasets.
No text outside JSON.
"""

_TRACING_BODY = (
    _CORE
    + _INCLUDE
    + "\n## Relationship Types (MUST use exactly one):\n"
    + RELATIONSHIP_TYPES
    + _EXCLUDE
    + _VERIFY
    + _NAME_EXTRACT
    + _OUTPUT
)

# ---------------------------------------------------------------------------
# HuggingFace README Analysis
# ---------------------------------------------------------------------------

HF_README_ANALYSIS_PROMPT = (
    """You are a dataset lineage expert. Identify the true source datasets \
used to construct the target dataset from its HuggingFace dataset card.

**Target Dataset:** {target_dataset}
**Dataset Card:**
{content}

**Source-specific notes (HuggingFace dataset card):**
- Pay special attention to structured sections like "Subsets", "Data Sources", \
"Dataset Composition" — these are high-confidence signals.
- Ignore YAML metadata headers; focus on the markdown body.
- **When the text contains an explicit HuggingFace dataset link** (e.g. https://huggingface.co/datasets/org/dataset-name or [text](https://huggingface.co/datasets/org/dataset-name)), you MUST use the canonical form **org/dataset-name** exactly as in that link. Do not alter it.
"""
    + _TRACING_BODY
)

# ---------------------------------------------------------------------------
# Blog Content Analysis
# ---------------------------------------------------------------------------

BLOG_CONTENT_ANALYSIS_PROMPT = (
    """You are a dataset lineage expert. Identify the true source datasets \
used to construct the target dataset from the following blog post.

**Target Dataset:** {target_dataset}
**Blog Content:**
{content}

**Source-specific notes (blog post):**
- Blogs mix narrative with data — distinguish selection language \
("we selected / chose / included / incorporated / added X") from comparison ("X performed best").
- Performance tables do NOT imply inclusion unless the text explicitly states selection.
- Watch for informal phrasing: "we grabbed data from", "pulled from", "built on top of".
- Non-selection mention does NOT imply inclusion.
- Performance comparison does NOT imply inclusion.
- **When a HuggingFace dataset link appears** (e.g. huggingface.co/datasets/org/name), use **org/name** exactly as in that link for the source name.
"""
    + _TRACING_BODY
)

# ---------------------------------------------------------------------------
# PDF Paper Analysis
# ---------------------------------------------------------------------------

PDF_DEEP_ANALYSIS_PROMPT = (
    """You are a dataset lineage expert. Identify the true source datasets \
used to construct the target dataset from the following paper excerpt.

**Target Dataset:** {target_dataset}
**Paper Content (excerpt):**
{full_text}

**Source-specific notes (academic paper):**
- Papers cite many datasets — most are NOT sources. Focus on "Data Construction" / \
"Dataset" / "Method" sections, not "Related Work" or "Experiments".
- Distinguish "we evaluate on X" / "we compare with X" (exclusion) from \
"we construct / derive / collect from X" (inclusion).
- Tables captioned "evaluation benchmarks" or "comparison results" are exclusions.
- Non-construction mention does NOT imply data usage.
- **When the paper gives a HuggingFace dataset URL or org/name**, use that exact form (**org/name**) as the source dataset name.
- Do NOT use license / legal / copyright / acknowledgement text as evidence.
- If a source table lists dataset names, the evidence should cite the source-table context or nearby introductory sentence saying these datasets are source data.
"""
    + _TRACING_BODY
)

# ---------------------------------------------------------------------------
# GitHub Repository Analysis
# ---------------------------------------------------------------------------

GITHUB_ANALYSIS_PROMPT = (
    """You are a dataset lineage expert. Identify the true source datasets \
used to construct the target dataset from the following GitHub repository.

**Repository:** {repo_url}
**Target Dataset:** {target_dataset}
**Repository README Content:**
{readme_content}

**Source-specific notes (GitHub repository):**
- You are analyzing the provided README text only. Do NOT assume access to files \
that are not shown here.
- If the README itself contains `load_dataset("org/name")`, a HuggingFace dataset URL, \
or an explicit `org/name`, use that exact form as the source dataset name.
- Distinguish training data sources from evaluation benchmarks in the README.
- Prefer explicit source wording over inference from general project description.
- **When a HuggingFace dataset URL or org/name appears**, use that exact format for the source name.
- NEVER include the target dataset itself (circular reference).
"""
    + _TRACING_BODY
)

# ---------------------------------------------------------------------------
# Link Extraction
# ---------------------------------------------------------------------------

LINK_EXTRACTION_PROMPT = """You are an expert at extracting relevant links from dataset documentation.

**Target Dataset:** {target_dataset}
(The links you extract must be about THIS specific dataset — papers that describe it, repos that create/host it, blogs that introduce it.)

**Content:**
{content}

## Extract (only links that directly describe or introduce the TARGET dataset above):
1. **Paper links** — Papers that describe or introduce THIS dataset.
2. **GitHub links** — Repos for processing, creating, or hosting THIS dataset.
3. **Blog links** — Posts about the creation or methodology of THIS dataset.

## Exclude:
- General ML/AI papers, libraries, or tutorials not about this dataset.
- Papers that only use (not describe) this dataset.
- General-purpose repos (e.g., huggingface/transformers).

Prefer official and authoritative sources. All URLs must be complete and valid.
It is normal to find no relevant links — always return valid JSON even with empty arrays.
When in doubt, exclude rather than include.

## Output
Return ONLY valid JSON:
{{
    "paper_links": [],
    "github_links": [],
    "blog_links": [],
    "extraction_confidence": 0.0,
    "extraction_notes": "Brief notes on findings"
}}

No text outside JSON.
"""

# ---------------------------------------------------------------------------
# Dataset Matching (simple name selection)
# ---------------------------------------------------------------------------

DATASET_MATCHING_PROMPT = """Given an original dataset name and search results, \
select the best match.

**Original Dataset:** {original_name}
**Context:**
{context_block}
**Search Results:** {candidates}

- Use the context to disambiguate short or ambiguous names.
- Prefer exact or near-exact name matches.
- Among similar candidates, prioritize higher download/view counts.
- Prefer well-known organizations.
- If no suitable match exists, return null.

Return ONLY valid JSON:
{{
    "selected_name": "org/dataset-name",
    "reasoning": "Brief explanation"
}}

If no suitable match exists:
{{
    "selected_name": null,
    "reasoning": "Why no suitable match exists"
}}
"""

# ---------------------------------------------------------------------------
# Dataset Name Inference
# ---------------------------------------------------------------------------

DATASET_NAME_INFERENCE_PROMPT = """You are an expert on HuggingFace datasets. \
The given name has no exact match. Infer 3 to 5 plausible canonical dataset names \
it might refer to (the input may already be normalized, e.g. citation suffixes removed).

**Name:** {dataset_name}
{context_block}

Requirements:
- Output 3 to 5 candidate names, one per line.
- Use common canonical forms (as used in papers, repos, or HuggingFace).
- Do NOT include organization prefixes (no "allenai/", "bigcode/").
- No explanations, numbering, or extra text.
- If context/evidence is provided above, use it to infer the most likely HuggingFace dataset identifiers.

Examples:
task_mmmlu → mmmlu
PrimeIntellect's SYNTHETIC-1 → SYNTHETIC-1

Candidate names:
"""

# ---------------------------------------------------------------------------
# Dataset Classification (non-multimodal)
# ---------------------------------------------------------------------------

DATASET_CLASSIFICATION_DETAILED_PROMPT = """You are a dataset classification expert. \
Classify the dataset into one or more categories.

**Name:** {dataset_name}
**Description:** {description}
**Source datasets:** {source_data_names}
**Evidence:** {evidence_section}

## Core Principle:
The dataset **name** often provides strong signals. Combine:
1. Semantic meaning of the name.
2. Explicit information in the description.
3. Naming conventions (e.g., "MathInstruct" → Math, "CodeBench" → Code, "SciQA" → Science).

Name-based evidence is allowed, but do NOT hallucinate or invent unsupported meanings.

## Categories (select from these ONLY):
1. **Math** — Mathematical reasoning, proofs, equations, competition problems, word problems.
2. **Code** — Programming, code generation, debugging, repositories, coding benchmarks.
3. **Science** — Scientific texts, research papers, scientific QA (physics, biology, chemistry, etc.).
4. **General** — General conversation, instruction following, broad multi-domain tasks.

## Rules:
- Multiple categories allowed.
- Use the **name** as a strong prior (e.g., "OpenR1-Math" → Math, "CodeBench" → Code).
- Use **source data** to refine: math sources → prefer Math; code sources → prefer Code.
- Use the **description** to confirm or adjust.
- If name and description conflict, prefer the description.
- If ambiguous, select most probable categories with lower confidence.

## Confidence Scoring:
- **0.9–1.0** — Name clearly indicates category AND description confirms.
- **0.7–0.9** — Name or description strongly suggests category; the other is neutral.
- **0.5–0.7** — Ambiguous signals; multiple categories plausible.

## Output
Return ONLY valid JSON. **categories MUST be a subset of ["Math", "Code", "Science", "General"]**:
{{
    "categories": ["Math"],
    "confidence": 0.85,
    "reasoning": "Brief explanation referencing name, description, and sources."
}}

No text outside JSON. No markdown fences.
"""

# ---------------------------------------------------------------------------
# Dataset Classification (multimodal)
# ---------------------------------------------------------------------------

DATASET_CLASSIFICATION_MULTIMODAL_PROMPT = """You are a dataset classification expert. \
Classify the multimodal dataset into one or more categories.

**Name:** {dataset_name}
**Description:** {description}
**Source datasets:** {source_data_names}
**Evidence:** {evidence_section}

## Core Principle:
The dataset **name** often provides strong signals. Combine:
1. Semantic meaning of the name.
2. Explicit information in the description.
3. Naming conventions (e.g., "OCR", "ChartQA", "DocVQA" → Infographic; "Spatial" → Spatial).

Name-based evidence is allowed, but do NOT hallucinate or invent unsupported meanings.

## Categories (select from these ONLY):
1. **General** — Broad multi-domain data, generic VQA/captioning/dialogue, general-purpose multimodal instruction following.
2. **Reasoning** — Multi-step reasoning, logic, planning, math-like reasoning, chain-of-thought problem solving.
3. **Spatial** — Spatial relationships, geometry, layout, navigation, 3D/2D positions, object relations, grounding, scene understanding.
4. **Infographic** — Charts, plots, tables, documents, slides, posters, forms, UI screenshots, OCR, document understanding.

## Calibration Examples (weak hints, always prioritize description):
- Zhiqiang007/MathV360K → Reasoning
- liuhaotian/LLaVA-Instruct-150K → Spatial, General, Reasoning
- MathLLMs/MM-MathInstruct → Reasoning
- HuggingFaceM4/FineVision → Spatial, General, Reasoning, Infographic
- MMR1/MMR1-SFT → Spatial, General, Reasoning, Infographic
- derek-thomas/ScienceQA → Reasoning

## Rules:
- Multiple categories allowed.
- Use the **name** as a strong prior (e.g., "ChartQA" → Infographic, "SpatialBench" → Spatial).
- Charts/tables/documents/OCR → strongly prefer **Infographic**.
- Spatial relations/grounding/navigation → strongly prefer **Spatial**.
- Multi-step inference/logical deduction → strongly prefer **Reasoning**.
- If name and description conflict, prefer the description.
- If ambiguous, select most probable categories with lower confidence.

## Confidence Scoring:
- **0.9–1.0** — Name clearly indicates category AND description confirms.
- **0.7–0.9** — Name or description strongly suggests category; the other is neutral.
- **0.5–0.7** — Ambiguous signals; multiple categories plausible.

## Output
Return ONLY valid JSON. **categories MUST be a subset of ["General", "Reasoning", "Spatial", "Infographic"]**:
{{
    "categories": ["General", "Reasoning"],
    "confidence": 0.85,
    "reasoning": "Brief explanation referencing name, description, and sources."
}}

No text outside JSON. No markdown fences.
"""

# ---------------------------------------------------------------------------
# README Summary
# ---------------------------------------------------------------------------

README_SUMMARY_PROMPT = """Generate a concise English summary (max 20 words) \
of the dataset's content and main use case.

**Dataset Name:** {dataset_name}
**README Content:**
{content}

Summary (max 20 words):
"""

# ---------------------------------------------------------------------------
# Dataset Type Determination
# ---------------------------------------------------------------------------

DATASET_TYPE_DETERMINATION_PROMPT = """Determine if the dataset is synthetic or manual/real.

Answer ONLY one word:
- "synthetic" — artificially generated, AI-created, LLM-produced
- "manual" — collected from real-world sources, human-annotated, naturally occurring

**Dataset Name:** {dataset_name}
**README Content:**
{content}

Answer:
"""

# ---------------------------------------------------------------------------
# Prompt dictionary for code access
# ---------------------------------------------------------------------------

PROMPTS = {
    'hf_readme_analysis': HF_README_ANALYSIS_PROMPT,
    'link_extraction': LINK_EXTRACTION_PROMPT,
    'blog_content_analysis': BLOG_CONTENT_ANALYSIS_PROMPT,
    'pdf_deep_analysis': PDF_DEEP_ANALYSIS_PROMPT,
    'github_analysis': GITHUB_ANALYSIS_PROMPT,
    'dataset_matching': DATASET_MATCHING_PROMPT,
    'dataset_name_inference': DATASET_NAME_INFERENCE_PROMPT,
    'dataset_classification_detailed': DATASET_CLASSIFICATION_DETAILED_PROMPT,
    'dataset_classification_multimodal': DATASET_CLASSIFICATION_MULTIMODAL_PROMPT,
    'readme_summary': README_SUMMARY_PROMPT,
    'dataset_type_determination': DATASET_TYPE_DETERMINATION_PROMPT,
}
