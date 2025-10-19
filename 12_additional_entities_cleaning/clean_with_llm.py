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

ANIMAL_STUDIES_GUIDE_DISEASE_LLM_ONLY = r"""

### OBJECTIVE ###
- IDENTIFY AND RETURN ONLY THE DISEASE/CONDITION(S) THAT THE STUDY INTENDS TO **IMPROVE/REVERSE/AMELIORATE** (THE THERAPEUTIC TARGET).
- RETURN OUTPUT STRICTLY USING **SCHEMA_DISEASE**.

### INPUT FORMAT ###
YOU WILL RECEIVE:
1) ABSTRACT: A SHORT TEXT DESCRIBING AN **ANIMAL** EXPERIMENT TESTING A NEW INTERVENTION.

### WHAT TO DO ###
From the ABSTRACT, **extract candidate disease/condition mentions directly from the text** (no pre-supplied list). Then apply the DECIDE rules below to keep only valid, relevant targets.

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
- **EACH "text" MUST BE COPIED VERBATIM FROM ABSTRACT** (DO NOT REWRITE, CANONICALIZE, OR INVENT).

### DECIDE (RULES) ###
For each **candidate span you found in the abstract**, decide whether to keep it using the following rules:

1. Validity
   - Keep only complete disease or condition terms.
   - Keep abbreviations (e.g., "AD", "PD", "MS") as **separate entities** if they are used in the abstract to refer to valid diseases/conditions.
   - Do not keep:
     • Partial tokens (e.g., "chronic", "tactile")
     • Pure adjectives or modifiers (e.g., "ischemic", "parkinsonian", "inflammatory")
     • Overly unspecific concepts (e.g., "brain damage", "neuronal damage", "neurodegeneration")

2. Relevance
   - Keep the disease(s)/condition(s) that are the **therapeutic target** of the study (i.e., what the intervention aims to improve/reverse/ameliorate).
   - Do not keep entities mentioned only as background, unrelated examples, or exclusion criteria.

3. Specificity
   - Keep both more general and more specific mentions of a disease when both appear in the study.
     • Example: If the text says “Alzheimer’s disease is a severe form of dementia… we tested a treatment for Alzheimer’s disease”, then keep both "Alzheimer’s disease" and "dementia".
     • Example: If the text says “we treated seizures… more concretely slow-wave burst seizure which is a form of focal seizure”, then keep "seizure", "slow-wave burst seizure", and "focal seizure".

4. Composite/linked conditions
   - If the target is described as a consequence of another disease, keep both.
     • Example: If the study treats "post-stroke seizure", then annotate **both** "seizure" and "stroke" as separate entities.

5. Final decision
   - Keep the entity only if it satisfies:
     • Valid disease/condition term, AND
     • Relevant (main therapeutic target), AND
     • Not filtered out as adjective, unspecific, or a pure descriptor.
"""

FEW_SHOT_EXAMPLES_DISEASE_LLM_ONLY = r"""
### FEW-SHOT EXAMPLES (EXTRACTION FROM SCRATCH) ###

Example 1
Abstract:  
Alzheimer’s disease is a severe form of dementia. In this study, we tested whether MKI-801 had a beneficial impact on memory impairment symptoms in Alzheimer’s disease mouse models.  

→ The therapeutic targets described are "Alzheimer’s disease", "dementia", and "memory impairment".  
→ Words like "chronic" or adjectives are not present.  
→ All entities are copied verbatim from the abstract.

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
In this study, we used a mouse model in which colitis was induced by DSS to test the therapeutic effects of a probiotic compound. Crohn’s disease and ulcerative colitis are common human inflammatory bowel diseases.

→ The therapeutic target actually studied in the animal experiment is "colitis".  
→ "Crohn’s disease" and "ulcerative colitis" are mentioned only as background examples (not treated in this study).  
→ Therefore, only "colitis" is kept.

Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "colitis" }
  ]
}
</json>


Example 3
Abstract:
We developed a novel compound that reduces neuronal loss and improves motor symptoms in a mouse model of Parkinson’s disease.

→ The disease being targeted is "Parkinson’s disease".  
→ Phrases like "neuronal loss" are descriptive, not valid disease entities.

Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "Parkinson’s disease" }
  ]
}
</json>


Example 4
Abstract:
The compound was tested in a rodent model of post-stroke seizure to evaluate its potential to alleviate seizure severity after ischemic stroke.

→ The text describes a composite target: "post-stroke seizure".  
→ Per rules, both "seizure" and "stroke" are annotated separately.

Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "seizure" },
    { "text": "stroke" }
  ]
}
</json>


Example 5
Abstract:
We tested an anti-inflammatory peptide in a rat model of multiple sclerosis (MS) to determine its effect on demyelination.

→ The target is "multiple sclerosis", which is also abbreviated as "MS" in the same abstract.  
→ Both should be kept as separate valid mentions.

Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "multiple sclerosis" },
    { "text": "MS" }
  ]
}
</json>
""".strip()

SYSTEM_DRUG = (
    "You are helping filter noisy NER drug recognition results."
)


SCHEMA_DRUG = """
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

ANIMAL_STUDIES_GUIDE_DRUG = r"""
### OBJECTIVE ###
- IDENTIFY AND RETURN ONLY THE DRUG/THERAPEUTIC INTERVENTION(S) THAT THE STUDY **ADMINISTERS OR TESTS AS TREATMENT INTERVENTIONS** IN ANIMAL SUBJECTS.
- RETURN OUTPUT STRICTLY USING **SCHEMA_DRUG**.

### INPUT FORMAT ###
YOU WILL RECEIVE:
1) ABSTRACT: A SHORT TEXT DESCRIBING AN **ANIMAL** EXPERIMENT TESTING A NEW INTERVENTION.
2) ENTITY_LIST: A CANDIDATE LIST OF DRUG-RELATED ENTITIES ALREADY IDENTIFIED BY NER.  

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

1) Definition / What counts as a DRUG here
   - Keep **active pharmacological substances** that are **explicitly administered/tested** in the animal study (small molecules, peptides, biologics, experimental codes, etc.).
   - Include brand names and generic names if either appears; keep them as separate entities if both appear.
   - Include co-treatments or adjuvants if they are **explicitly administered** as part of the intervention.

2) Form and dosing details
   - Keep only the **substance name**, not the dose, route, schedule, or formulation.
   - Exclude pure formulation words and vehicles (e.g., "saline", "vehicle", "corn oil", "CMC") unless pharmacologically active.

3) Specificity and classes
   - Keep the **most specific drug name** (e.g., "fingolimod", "ibuprofen").
   - **Do NOT keep pharmacologic class phrases** (e.g., "beta-blocker", "NSAID", "SSRI", "monoclonal antibody") if a specific drug name for the same intervention appears.
   - Keep a class term only if:
     • it is the only information available for the administered intervention, or  
     • the class itself is administered as the intervention, or  
     • it refers to a distinct intervention separate from any named drug.

4) Negation and controls
   - Keep drug mentions even if negated when referring to the intervention/control design.
   - Exclude drugs mentioned only as unrelated background.

5) Conjugates, combinations, and linked forms
   - For conjugates/complexes, keep component drug names as separate entities when present.
   - For combinations (e.g., "acetaminophen-codeine"), keep each identifiable component as a separate entity.

6) Idiomatic multiword drug phrases
   - If a multiword phrase is idiomatic and pharmacologically defining, keep the full phrase.
   - Do not include non-defining adjectives unless part of such a term.

7) Abbreviations and codes
   - Keep experimental codes/abbreviations (e.g., "MK-801", "AZD1234", "L-NAME") only if they denote an administered substance.

8) Exclusions (filter out)
   - Partial tokens ("selective" alone, "agonist" alone) unless covered by Rule 6.
   - Routes/schedules/doses/units.
   - Delivery devices/non-active carriers (e.g., "liposome") unless the payload drug itself appears (keep the payload).
   - Pure targets without drug identity (e.g., "PPAR" alone).

9) Relevance
   - Keep drugs **administered to animals** (treatment or comparator, including standards).
   - Exclude drugs mentioned only as unrelated background.

10) Final decision checklist
   Keep the entity only if:
   - It denotes an **active substance** (or an allowed specific class term),
   - It is **relevant** to the interventions/controls,
   - It is **not** filtered out by the exclusions above.

11) Alias collapse & name precedence
   - When multiple strings clearly refer to the **same substance/intervention**, keep only one according to the following priority:
     **Generic/INN or common name > Brand name > Experimental/code name > Chemical/systematic name > Class/descriptor.**
   - If both a specific drug name and a class/descriptor refer to the same intervention, keep only the specific drug name.
   - Apply this only among entities that clearly refer to the same administered drug; otherwise, keep distinct drugs separately.

12) Explicit administration criterion
   - Retain entities **only if they are explicitly described as administered, injected, treated, or given to animals** as part of an intervention or control condition.
   - Exclude compounds mentioned only in **mechanistic, cellular, or molecular contexts** without clear in vivo administration, such as:
     • “FGFR1 was inhibited by PD166866”  
     • “cells were pre-treated with inhibitor X in vitro”  
     • “signaling was blocked using compound Y”
   - Keep only substances where the text indicates actual **treatment of animals** (e.g., “mice were treated with…”, “rats received…”, “administered i.p.”, “given orally”).
"""

FEW_SHOT_EXAMPLES_DRUG = r"""
### FEW-SHOT EXAMPLES (ADAPTED TO SCHEMA) ###

Example 1 — Specific name present → drop class
Abstract:
In this study, we tested MKI-801 (20 mg, p.o.) on learning behavior in mice. MKI-801 is an NMDA receptor antagonist.
Entities:
- MKI-801
- NMDA receptor antagonist
- 20 mg
- p.o.
Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "MKI-801" }
  ]
}
</json>

---

Example 2 — Alias collapse (generic > code)
Abstract:
The treated group received fingolimod (FTY720) daily for seven days, controls received vehicle.
Entities:
- fingolimod
- FTY720
- vehicle
Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "fingolimod" }
  ]
}
</json>

---

Example 3 — Class kept only when no specific drug is named
Abstract:
Animals were administered an SSRI for 14 days to assess behavioral effects.
Entities:
- SSRI
- 14 days
Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "SSRI" }
  ]
}
</json>

---

Example 4 — Conjugate/complex → keep component drugs
Abstract:
To study oxidative stress, we synthesized an ibuprofen glutathione conjugate and compared it with ibuprofen alone.
Entities:
- ibuprofen glutathione conjugate
- ibuprofen
- glutathione
- oxidative stress
Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "ibuprofen" },
    { "text": "glutathione" }
  ]
}
</json>

---

Example 5 — Explicit administration required (exclude mechanistic-only mentions)
Abstract:
FGFR1 was inhibited by PD166866 in cultured oligodendrocytes. Mice were treated with IFNβ-1a (400 ng/mL).
Entities:
- PD166866
- IFNβ-1a
- 400 ng/mL
Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "IFNβ-1a" }
  ]
}
</json>

---

Example 6 — Alias collapse (generic/common > chemical/specific form)
Abstract:
We administered 1,25-dihydroxyvitamin D or Ergocalciferol; both are forms of VITAMIN D.
Entities:
- 1,25-dihydroxyvitamin D
- Ergocalciferol
- VITAMIN D
Expected:
ANSWER: <json>
{
  "entities": [
    { "text": "VITAMIN D" }
  ]
}
</json>

---

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

def build_prompt_llm_extractor(abstract, SYSTEM, ANIMAL_STUDIES_GUIDE, SCHEMA, FEW_SHOT_EXAMPLES):
    """
    Build a strict instruction prompt specialized for preclinical animal studies.
    - Designed for extracting disease/condition entities directly from the abstract (no entity list provided).
    - Preserves schema structure and <json> wrapper.
    - Enforces strict formatting and return constraints.
    """

    return (
        f"SYSTEM:\n{SYSTEM}\n\n"
        "USER:\n"
        f"{ANIMAL_STUDIES_GUIDE}\n\n"
        "### INPUT FORMAT ###\n"
        "- YOU WILL RECEIVE:\n"
        "  - A BIOMEDICAL ABSTRACT DESCRIBING AN ANIMAL STUDY.\n"
        "### OUTPUT FORMAT ###\n"
        "- RETURN ONLY A JSON OBJECT INSIDE <json> AND </json> TAGS, MATCHING THIS SCHEMA:\n"
        f"{SCHEMA}\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"NEW Abstract:\n<<<{abstract}>>>\n\n"
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
            #prompt_id=prompt_id      # (if your cleaner uses it)
        )
        results.append(val)

        if (idx + 1) % args.checkpoint_every == 0:
            df[new_col_name] = pd.Series(results, index=df.index[:len(results)])
            ckpt_path = os.path.join(inter_dir, f"{os.path.basename(args.output)}.{entity_type}.part.csv")
            df_ckpt = df.drop(columns=["Text"], errors="ignore")
            df_ckpt.to_csv(ckpt_path, index=False)
            print(f"[CKPT] Saved {idx+1} rows → {ckpt_path}")

    df[new_col_name] = pd.Series(results, index=df.index)

    # Fallback if empty
    if fallback_col in df.columns:
        mask_empty = df[new_col_name].fillna("").str.strip().eq("")
        df.loc[mask_empty, new_col_name] = df.loc[mask_empty, fallback_col]
    else:
        print(f"[WARN] Fallback column missing: {fallback_col}")

    df = df.drop(columns=["Text"], errors="ignore")
    df.to_csv(args.output, index=False)
    print(f"[DONE] Saved: {args.output}")

if __name__ == "__main__":
    main()
