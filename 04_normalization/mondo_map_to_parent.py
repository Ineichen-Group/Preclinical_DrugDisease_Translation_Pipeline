import argparse
from pathlib import Path
from collections import deque
from typing import Tuple, Dict, Set, Iterable

import pandas as pd
import pronto


# ------------------ helpers ------------------

def parse_mondo_ids(series: pd.Series) -> Set[str]:
    """
    Extract all valid MONDO IDs from a pandas Series containing
    pipe-separated MONDO identifiers.

    - Skips empty values and '-1'
    - Used to build the global set of MONDO terms observed in the dataset

    Parameters
    ----------
    series : pd.Series
        Column containing pipe-separated MONDO IDs.

    Returns
    -------
    Set[str]
        Unique MONDO IDs found in the column.
    """
    
    ids = set()
    for cell in series.fillna("").astype(str):
        for tid in cell.split("|"):
            if tid and tid != "-1":
                ids.add(tid)
    return ids


def build_ancestor_and_distance_maps(
    ontology: pronto.Ontology,
    all_mondo_ids: Set[str],
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], int]]:
    """
    Precompute ontology traversal structures for fast lookup.

    For each MONDO term in the dataset:
      - ancestor_map: all transitive MONDO ancestors
      - distance_map: number of is_a hops from term → ancestor

    This avoids repeated ontology traversal during row-level processing.

    Parameters
    ----------
    ontology : pronto.Ontology
        Loaded MONDO ontology.
    all_mondo_ids : Set[str]
        MONDO IDs observed across clinical + preclinical data.

    Returns
    -------
    ancestor_map : Dict[str, Set[str]]
        term_id → set of ancestor MONDO IDs.
    distance_map : Dict[(str, str), int]
        (child_id, ancestor_id) → hop distance.
    """
    
    ancestor_map = {}
    distance_map = {}

    for mid in all_mondo_ids:
        try:
            term = ontology[mid]
        except KeyError:
            continue

        visited = {mid: 0}
        queue = deque([(term, 0)])

        while queue:
            node, dist = queue.popleft()
            for parent in node.superclasses(distance=1, with_self=False):
                pid = parent.id
                if not pid.startswith("MONDO:"):
                    continue
                if pid not in visited:
                    visited[pid] = dist + 1
                    queue.append((parent, dist + 1))

        ancestor_map[mid] = set(visited) - {mid}
        for pid, dist in visited.items():
            if pid != mid:
                distance_map[(mid, pid)] = dist

    return ancestor_map, distance_map


def compute_mondo_term_metrics(
    ontology: pronto.Ontology,
    root_id: str,
    term_ids: Iterable[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Compute structural specificity metrics for MONDO terms.

    Metrics:
      1. depth_to_root: shortest is_a distance to the chosen root term
      2. desc_count: total number of transitive subclasses

    Used to filter out overly generic ontology terms.

    Parameters
    ----------
    ontology : pronto.Ontology
        Loaded MONDO ontology.
    root_id : str
        MONDO ID treated as ontology root.
    term_ids : Iterable[str]
        MONDO terms to score.

    Returns
    -------
    depth_to_root : Dict[str, int]
        MONDO ID → depth from root.
    desc_count : Dict[str, int]
        MONDO ID → number of descendants.
    """
    
    depth_to_root = {}
    desc_count = {}

    root = ontology[root_id]

    for tid in term_ids:
        try:
            term = ontology[tid]
        except KeyError:
            continue

        # depth to root
        visited = {tid: 0}
        queue = deque([(term, 0)])
        d_root = None

        while queue:
            node, dist = queue.popleft()
            if node.id == root.id:
                d_root = dist
                break
            for parent in node.superclasses(distance=1, with_self=False):
                pid = parent.id
                if pid not in visited:
                    visited[pid] = dist + 1
                    queue.append((parent, dist + 1))

        depth_to_root[tid] = d_root if d_root is not None else float("inf")

        # descendant count
        descendants = set()
        queue = deque([term])

        while queue:
            node = queue.popleft()
            for child in node.subclasses(distance=1, with_self=False):
                cid = child.id
                if cid not in descendants:
                    descendants.add(cid)
                    queue.append(child)

        desc_count[tid] = len(descendants)

    return depth_to_root, desc_count

def save_stats(mapped_to_parent: dict, out_path: str):
    """
    Save mapping statistics for dataset → parent assignments.

    Each row represents one (entity → parent) mapping and records
    how many original mentions were collapsed under it.

    Parameters
    ----------
    mapped_to_parent : dict
        parent_key -> list of child values (e.g. mentions or IDs)
    out_path : str
        Output CSV path.
    """
    rows = []
    for key, values in mapped_to_parent.items():
        rows.append({
            "mapping": key,
            "entity": key.split("(")[0],
            "parent": key.split("(")[1].rstrip(")") if "(" in key else "",
            "ner_count": len(values),
            "children_values": values,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("ner_count", ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)

# ------------------ core logic ------------------

def assign_nearest_dataset_parents(
    df: pd.DataFrame,
    ontology: pronto.Ontology,
    all_mondo_ids: Set[str],
    ancestor_map: Dict[str, Set[str]],
    depth_to_root: Dict[str, int],
    desc_count: Dict[str, int],
    distance_map: Dict[Tuple[str, str], int],
    id_column: str,
    min_depth: int,
    max_desc: int,
    stats_dict: dict | None = None,
) -> pd.DataFrame:
    """
    For each MONDO term in each row, select the nearest valid
    ancestor that also appears elsewhere in the dataset.

    Selection criteria:
      - ancestor appears in dataset
      - depth_to_root >= min_depth
      - descendant count < max_desc
      - minimal is_a distance from child

    Adds two columns:
      - nearest_dataset_parent_mondo
      - nearest_dataset_parent_label

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (clinical or preclinical).
    ontology : pronto.Ontology
        Loaded MONDO ontology.
    all_mondo_ids : Set[str]
        All MONDO IDs observed in the dataset.
    ancestor_map : dict
        Precomputed ancestor relationships.
    depth_to_root : dict
        Depth metric for MONDO terms.
    desc_count : dict
        Descendant count metric.
    distance_map : dict
        Precomputed hop distances.
    id_column : str
        Column containing pipe-separated MONDO IDs.
    min_depth : int
        Minimum allowed depth from root.
    max_desc : int
        Maximum allowed descendant count.

    Returns
    -------
    pd.DataFrame
        Copy of df with nearest parent columns added.
    """
    
    parent_ids = []
    parent_labels = []

    for _, row in df.iterrows():
        input_ids = [
            tid for tid in str(row[id_column]).split("|")
            if tid and tid != "-1"
        ]

        row_parents = []
        row_labels = []

        for child_id in input_ids:
            best_ancestor = None
            best_distance = None

            for ancestor_id in ancestor_map.get(child_id, []):
                if ancestor_id not in all_mondo_ids:
                    continue

                depth = depth_to_root.get(ancestor_id, float("inf"))
                if not (min_depth <= depth < float("inf")):
                    continue

                if desc_count.get(ancestor_id, 0) >= max_desc:
                    continue

                dist = distance_map.get((child_id, ancestor_id), float("inf"))

                # nearest ancestor = smallest distance
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_ancestor = ancestor_id

            if best_ancestor:
                row_parents.append(best_ancestor)
                row_labels.append(ontology[best_ancestor].name)
                if stats_dict is not None:
                    key = f"{ontology[best_ancestor].name}({best_ancestor})"
                    stats_dict.setdefault(key, []).append(child_id)
            else:
                row_parents.append("-1")
                row_labels.append("-1")

        parent_ids.append("|".join(row_parents) if row_parents else "-1")
        parent_labels.append("|".join(row_labels) if row_labels else "-1")

    out = df.copy()
    out["nearest_dataset_parent_mondo"] = parent_ids
    out["nearest_dataset_parent_label"] = parent_labels
    return out


def merge_original_and_parent_mondo(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Merge original MONDO annotations with inferred dataset parents,
    keeping only unique MONDO IDs (order-preserving).

    Rules:
      - Preserve original MONDO IDs first
      - Append parent MONDO terms only if not already present
      - Ensure ID–label alignment
      - Remove duplicates across originals and parents

    Adds:
      - merged_mondo_termid
      - merged_mondo_label
    """
    merged_ids = []
    merged_labels = []

    for _, row in df.iterrows():
        # Original entries
        orig_ids = str(row[id_col]).split("|")
        orig_labels = str(row[label_col]).split("|")

        # Parent entries
        parent_ids = str(row["nearest_dataset_parent_mondo"]).split("|")
        parent_labels = str(row["nearest_dataset_parent_label"]).split("|")

        seen = set()
        mids = []
        mlabs = []

        # 1) add originals (in order)
        for oid, olab in zip(orig_ids, orig_labels):
            if oid not in seen:
                seen.add(oid)
                mids.append(oid)
                mlabs.append(olab)

        # 2) add parents (if valid and unseen)
        for pid, plab in zip(parent_ids, parent_labels):
            if pid != "-1" and pid not in seen:
                seen.add(pid)
                mids.append(pid)
                mlabs.append(plab)

        merged_ids.append("|".join(mids))
        merged_labels.append("|".join(mlabs))

    df = df.copy()
    df["merged_mondo_termid"] = merged_ids
    df["merged_mondo_label"] = merged_labels
    return df



# ------------------ main ------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--clinical_input", required=True)
    p.add_argument("--preclinical_input", required=True)
    p.add_argument("--clinical_output", required=True)
    p.add_argument("--preclinical_output", required=True)

    p.add_argument("--ontology_path", required=True)
    p.add_argument("--root_id", default="MONDO:0000001")

    p.add_argument("--id_column", default="disease_mondo_termid")
    p.add_argument("--label_column", default="disease_term_mondo_norm")

    p.add_argument("--min_depth", type=int, default=5)
    p.add_argument("--max_desc", type=int, default=500)

    args = p.parse_args()

    # Load inputs
    df_clinical = pd.read_csv(args.clinical_input, dtype=str)
    df_preclinical = pd.read_csv(args.preclinical_input, dtype=str)

    ontology = pronto.Ontology(args.ontology_path)

    # Build joint MONDO universe
    ids_clinical = parse_mondo_ids(df_clinical[args.id_column])
    ids_preclinical = parse_mondo_ids(df_preclinical[args.id_column])
    all_mondo_ids = ids_clinical | ids_preclinical

    ancestor_map, distance_map = build_ancestor_and_distance_maps(
        ontology, all_mondo_ids
    )

    candidates = set(ancestor_map.keys())
    for s in ancestor_map.values():
        candidates |= s

    depth_to_root, desc_count = compute_mondo_term_metrics(
        ontology,
        args.root_id,
        candidates,
    )
    mapped_to_parent_clinical = {}
    mapped_to_parent_preclinical = {}
    # Clinical
    df_c = assign_nearest_dataset_parents(
        df_clinical,
        ontology,
        all_mondo_ids,
        ancestor_map,
        depth_to_root,
        desc_count,
        distance_map,
        args.id_column,
        args.min_depth,
        args.max_desc,
        stats_dict=mapped_to_parent_clinical,

    )
    df_c = merge_original_and_parent_mondo(
        df_c,
        args.id_column,
        args.label_column,
    )
    Path(args.clinical_output).parent.mkdir(parents=True, exist_ok=True)
    df_c.to_csv(args.clinical_output, index=False)

    # Preclinical
    df_p = assign_nearest_dataset_parents(
        df_preclinical,
        ontology,
        all_mondo_ids,
        ancestor_map,
        depth_to_root,
        desc_count,
        distance_map,
        args.id_column,
        args.min_depth,
        args.max_desc,
        stats_dict=mapped_to_parent_preclinical,

    )
    df_p = merge_original_and_parent_mondo(
        df_p,
        args.id_column,
        args.label_column,
    )
    Path(args.preclinical_output).parent.mkdir(parents=True, exist_ok=True)
    df_p.to_csv(args.preclinical_output, index=False)
    
    stats_dir = Path("./data/mondo/")
    stats_dir.mkdir(parents=True, exist_ok=True)

    save_stats(
        mapped_to_parent_clinical,
        stats_dir / "mondo_mapped_to_parents_clinical_stats.csv",
    )

    save_stats(
        mapped_to_parent_preclinical,
        stats_dir / "mondo_mapped_to_parents_preclinical_stats.csv",
    )



if __name__ == "__main__":
    main()
