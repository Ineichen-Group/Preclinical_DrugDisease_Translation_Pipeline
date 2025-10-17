# clean_chunk.py
import os
import argparse
import pandas as pd
from tqdm import tqdm
import os, re, json
from typing import List, Dict, Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from vllm import LLM, SamplingParams


SYSTEM = (
    "You are helping filter noisy NER disease recognition results."
)

SCHEMA_DISEASE = """
<json>
{
  "entities": [
    {
      "text": "<entity string>"
    }
  ]
}
</json>
""".strip()

ANIMAL_STUDIES_GUIDE_DISEASE = r"""

### OBJECTIVE ###
- IDENTIFY AND RETURN ONLY THE DISEASE/CONDITION(S) THAT THE STUDY INTENDS TO **IMPROVE/REVERSE/AMELIORATE** (THE THERAPEUTIC TARGET).
- RETURN OUTPUT STRICTLY USING **SCHEMA_DISEASE**.

### INPUT FORMAT ###
YOU WILL RECEIVE:
1) ABSTRACT: A SHORT TEXT DESCRIBING AN **ANIMAL** EXPERIMENT TESTING A NEW INTERVENTION.
2) ENTITY_LIST: A CANDIDATE LIST OF DISEASE-RELATED ENTITY ALREADY IDENTIFIED BY NER.  

### OUTPUT FORMAT (STRICT) ###
RETURN EXACTLY:
ANSWER: <json>
{
  "entities": [
    { "text": "<entity string>" }
  ]
}
</json>
- RETURN ZERO, ONE, OR MULTIPLE ENTITIES.
- **EACH "text" MUST BE COPIED VERBATIM FROM ENTITY_LIST** (DO NOT REWRITE, CANONICALIZE, OR INVENT).

### DECIDE (RULES) ###
For each entity in ENTITY_LIST, decide whether to keep it using the following rules:

1. Validity
   - Keep only complete disease or condition terms.
   - Keep abbreviations (e.g., "AD", "PD", "MS") as **separate entities** if they are used in the abstract to refer to valid diseases/conditions.
   - Do not keep:
     • Partial tokens (e.g., "chronic", "tactile")
     • Pure adjectives or modifiers (e.g., "ischemic", "parkinsonian", "inflammatory")
     • Overly unspecific concepts (e.g., "brain damage", "neuronal damage", "neurodegeneration")

2. Relevance
   - Keep the disease(s)/condition(s) that are the therapeutic target of the study.
   - Do not keep entities mentioned only as background, unrelated examples, or exclusion criteria.

3. Specificity
   - Keep both more general and more specific mentions of a disease when both appear in the study.
     • Example: If the text says “Alzheimer’s disease is a severe form of dementia… we tested a treatment for Alzheimer’s disease”, then both "Alzheimer’s disease" and "dementia" should be kept.
     • Example: If the text says “we treated seizures… more concretely slow-wave burst seizure which is a form of focal seizure”, then "seizure", "slow-wave burst seizure", and "focal seizure" should all be kept.

4. Composite/linked conditions
   - If the target is described as a consequence of another disease, keep both.
     • Example: If the study treats "post-stroke seizure", then annotate "seizure" and "stroke" separately.

5. Final decision
   - Keep the entity only if it satisfies:
     • Valid disease/condition term, AND
     • Relevant (main therapeutic target), AND
     • Not filtered out as adjective, unspecific or as a pure descriptor.

"""

FEW_SHOT_EXAMPLES_DISEASE = r"""
### FEW-SHOT EXAMPLES (ADAPTED TO SCHEMA) ###

Example 1
Abstract:  
Alzheimer’s disease is a severe form of dementia. In this study, we tested whether MKI-801 had a beneficial impact on memory impairment symptoms in Alzheimer’s disease mouse models.  

Entities: 
- Alzheimer’s disease  
- dementia  
- chronic  
- memory impairment  

Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "Alzheimer’s disease" },
    { "text": "dementia" },
    { "text": "memory impairment" }
  ]
}
</json>

Example 2
Abstract:
"In this study, we used a mouse model to induce colitis with DSS to study the effect of probiotic Y. Crohn’s disease and ulcerative colitis are common forms of inflammatory bowel disease in humans."

Entities:
- colitis
- Crohn’s disease
- ulcerative colitis
- inflammatory bowel disease

Expected:
ANSWER: <json>
{
  "entities": [
    {
      "text": "colitis"
      
    }
  ]
}
</json>
""".strip()


def build_prompt(abstract, entities, SYSTEM, ANIMAL_STUDIES_GUIDE, SCHEMA, FEW_SHOT_EXAMPLES):
    """
    Build a strict instruction prompt specialized for preclinical animal studies.
    - Preserves your schema and <json> wrapper.
    - Enforces ordering, no entity changes, ≤12-word evidence quotes.
    """
    ents = "\n".join(f"- {e['text']}" for e in entities)

    return (
        f"SYSTEM:\n{SYSTEM}\n\n"
        "USER:\n"
        f"{ANIMAL_STUDIES_GUIDE}\n\n"
        "### INPUT FORMAT ###\n"
        "- YOU WILL RECEIVE:\n"
        "  - A BIOMEDICAL ABSTRACT.\n"
        "  - A LIST OF DISEASE ENTITIES FOUND IN THAT ABSTRACT.\n\n"
        "### OUTPUT FORMAT ###\n"
        "- RETURN ONLY A JSON OBJECT INSIDE <json> AND </json> TAGS, MATCHING THIS SCHEMA:\n"
        f"{SCHEMA}\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"NEW Abstract:\n<<<{abstract}>>>\n\n"
        "Entities:\n<<<\n"
        f"{ents}\n>>>\n"
        "Return your answer strictly as JSON inside <json>...</json>, nothing else:\n<<<\n"
        
    )


sampling = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=2000,
    stop=["</json>"],   # <- important
)

def extract_json(text: str):
    # 1. Preferred case: inside <json>...</json>
    m = re.search(r"<json>\s*(\{.*\})\s*(?:</json>)?", text, flags=re.S)
    if m:
        #print(m.group(1))
        return json.loads(m.group(1))

    # 2. Fenced code block: ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        return json.loads(m.group(1))

    # 3. Any last JSON-looking object in the string
    return _parse_last_json_anywhere(text)


def _parse_last_json_anywhere(text: str):
    """
    Try to grab the last {...} block from the text and parse it as JSON.
    Useful if LLM prepends explanations or stray text.
    """
    matches = list(re.finditer(r"\{.*\}", text, flags=re.S))
    if not matches:
        raise ValueError(f"No JSON object found in: {text}")
    for m in reversed(matches):  # last one usually the valid one
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from: {text}")

# IMPORTANT: you must have this in scope/importable
# from your_module import apply_llm_cleanup

def apply_llm_cleanup(row, entities_col_name, entity_type, max_retries=3, llm_only=False, llm=None):
    abstract = row['Text']
    entities_str = row[entities_col_name]

    # Handle NaN, None, or empty string
    if not entities_str or str(entities_str).strip() == "" or pd.isna(entities_str):
        return entities_str

    # Build fake entities from the pipe-separated string
    entities = [
        {"text": term.strip(), "type": entity_type}
        for term in str(entities_str).split("|") if term.strip()
    ]
    original_entities = {ent["text"] for ent in entities}
    if (len(original_entities) == 1) and (not llm_only):
        entity_txt = original_entities.pop()
        print(f"single entity, returning: {entity_txt}")
        return entity_txt
    

    # Build prompt
    if entity_type.upper() == 'DISEASE':
        if llm_only:
            print("Building disease prompt for LLM only extraction...")
            prompt = build_prompt_llm_extractor(
                abstract,
                SYSTEM, ANIMAL_STUDIES_GUIDE_DISEASE_LLM_ONLY, SCHEMA_DISEASE, FEW_SHOT_EXAMPLES_DISEASE_LLM_ONLY
            )
        else:
            print("Building disease prompt...")
            prompt = build_prompt(
                abstract, entities,
                SYSTEM, ANIMAL_STUDIES_GUIDE_DISEASE, SCHEMA_DISEASE, FEW_SHOT_EXAMPLES_DISEASE
            )
    else:
        print("Building drug prompt...")
        prompt = build_prompt(
            abstract, entities,
            SYSTEM_DRUG, ANIMAL_STUDIES_GUIDE_DRUG, SCHEMA_DRUG, FEW_SHOT_EXAMPLES_DRUG
        )

    # Try extraction with retries
    for attempt in range(1, max_retries + 1):
        try:
            out = llm.generate([prompt], sampling)
            text = out[0].outputs[0].text
            print(text)
            data = extract_json(text)  # may raise

            primary_targets = [
                ent["text"] for ent in data.get("entities", [])
                #if ent.get("role") == "primary_target"
            ]

            # ✅ Validation: all extracted must be in the original set
            if llm_only or all(pt in original_entities for pt in primary_targets):
                result = "|".join(primary_targets)
                print("original entities: ", original_entities)
                print(f"output: {result}")
                return result
            else:
                print(f"⚠️ attempt {attempt}: extracted entities not in original list → {primary_targets}")
                if attempt == max_retries:
                    print("❌ Giving up after max retries. Returning original entities.")
                    return entities_str
                continue

        except Exception as e:
            print(f"❌ attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("⚠️ Giving up after max retries. Returning original entities.")
                return entities_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt-id", default="prompt1_32B_FS_LLM_ONLY")
    ap.add_argument("--target-col", default="conditions", choices=["conditions","interventions"])
    ap.add_argument("--entity-type", default="DISEASE")
    ap.add_argument("--llm-only", action="store_true")
    ap.add_argument("--entities-col-name", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--intermediate-dir", default=None)
    ap.add_argument("--force", action="store_true")

    # vLLM / model args
    ap.add_argument("--model-dir", default="/shares/animalwelfare.crs.uzh/llms/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--tp", type=int, default=1)               # tensor_parallel_size
    ap.add_argument("--max-len", type=int, default=8192)       # max_model_len
    ap.add_argument("--dtype", default=None)                   # e.g. "bfloat16" or "float16"
    args = ap.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"[SKIP] Output exists: {args.output}")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    inter_dir = args.intermediate_dir or os.path.join(os.path.dirname(args.output), "intermediate")
    os.makedirs(inter_dir, exist_ok=True)

    # ---- Load vLLM offline
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    llm_kwargs = dict(
        model=args.model_dir,
        tokenizer=args.model_dir,
        trust_remote_code=True,
        max_model_len=args.max_len,
        tensor_parallel_size=args.tp,
    )
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype

    print(f"[LLM] Loading from {args.model_dir} (tp={args.tp}, max_len={args.max_len}, dtype={args.dtype})")
    llm = LLM(**llm_kwargs)

    # ---- Load data
    print(f"[INFO] Reading: {args.input}")
    df = pd.read_csv(args.input)

    prompt_id = args.prompt_id
    target_col = args.target_col
    entity_type = args.entity_type
    llm_only = args.llm_only

    new_col_name = (
        f"unique_{target_col}_LLM_extractor_{prompt_id}"
        if llm_only else
        f"unique_{target_col}_biolinkbert_llm_clean_{prompt_id}"
    )
    fallback_col = args.entities_col_name or f"unique_{target_col}_biolinkbert"

    tqdm.pandas(desc=f"Cleaning {target_col} ({'LLM-only' if llm_only else 'LLM-clean'})")

    # Process with checkpoints
    results = []
    for idx, row in df.iterrows():
        val = apply_llm_cleanup(
            row,
            entities_col_name=fallback_col,
            entity_type=entity_type,
            llm_only=llm_only,
            llm=llm,                 # <<<<<< pass the vLLM instance here
            prompt_id=prompt_id      # (if your cleaner uses it)
        )
        results.append(val)

        if (idx + 1) % args.checkpoint_every == 0:
            df[new_col_name] = pd.Series(results, index=df.index[:len(results)])
            ckpt_path = os.path.join(inter_dir, f"{os.path.basename(args.output)}.part.csv")
            df.to_csv(ckpt_path, index=False)
            print(f"[CKPT] Saved {idx+1} rows → {ckpt_path}")

    df[new_col_name] = pd.Series(results, index=df.index)

    # Fallback if empty
    if fallback_col in df.columns:
        mask_empty = df[new_col_name].fillna("").str.strip().eq("")
        df.loc[mask_empty, new_col_name] = df.loc[mask_empty, fallback_col]
    else:
        print(f"[WARN] Fallback column missing: {fallback_col}")

    df.to_csv(args.output, index=False)
    print(f"[DONE] Saved: {args.output}")

if __name__ == "__main__":
    main()
