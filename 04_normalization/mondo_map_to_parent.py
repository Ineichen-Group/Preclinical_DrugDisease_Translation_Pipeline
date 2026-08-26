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

    Skips empty values and '-1'.
    """
    ids = set()

    for cell in series.fillna("").astype(str):
        for tid in cell.split("|"):
            tid = tid.strip()

            if tid and tid != "-1":
                ids.add(tid)

    return ids


def build_ancestor_and_distance_maps(
    ontology: pronto.Ontology,
    all_mondo_ids: Set[str],
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], int]]:
    """
    Precompute transitive MONDO ancestors and child -> ancestor distances.
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

            for parent in node.superclasses(
                distance=1,
                with_self=False,
            ):
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
    Compute:
      - shortest distance to MONDO root
      - total number of descendants
    """
    depth_to_root = {}
    desc_count = {}

    root = ontology[root_id]

    for tid in term_ids:
        try:
            term = ontology[tid]
        except KeyError:
            continue

        # Depth to root
        visited = {tid: 0}
        queue = deque([(term, 0)])
        d_root = None

        while queue:
            node, dist = queue.popleft()

            if node.id == root.id:
                d_root = dist
                break

            for parent in node.superclasses(
                distance=1,
                with_self=False,
            ):
                pid = parent.id

                if pid not in visited:
                    visited[pid] = dist + 1
                    queue.append((parent, dist + 1))

        depth_to_root[tid] = (
            d_root if d_root is not None else float("inf")
        )

        # Descendant count
        descendants = set()
        queue = deque([term])

        while queue:
            node = queue.popleft()

            for child in node.subclasses(
                distance=1,
                with_self=False,
            ):
                cid = child.id

                if cid not in descendants:
                    descendants.add(cid)
                    queue.append(child)

        desc_count[tid] = len(descendants)

    return depth_to_root, desc_count


def save_stats(mapped_to_parent: dict, out_path: str):
    """
    Save statistics for child -> parent mappings.
    """
    rows = []

    for key, values in mapped_to_parent.items():
        rows.append(
            {
                "mapping": key,
                "entity": key.split("(")[0],
                "parent": (
                    key.split("(")[1].rstrip(")")
                    if "(" in key
                    else ""
                ),
                "ner_count": len(values),
                "children_values": values,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            "ner_count",
            ascending=False,
        ).reset_index(drop=True)

    Path(out_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(out_path, index=False)


# ------------------ core logic ------------------

def assign_nearest_dataset_parents(
    df: pd.DataFrame,
    ontology: pronto.Ontology,
    candidate_parent_ids: Set[str],
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
    For each MONDO term in each row, select the nearest valid ancestor
    that also occurs in the candidate-parent universe.

    Candidate parents may originate from:
      - historical clinical data
      - historical preclinical data
      - new preclinical data
    """
    parent_ids = []
    parent_labels = []

    for _, row in df.iterrows():
        input_ids = [
            tid.strip()
            for tid in str(row[id_column]).split("|")
            if tid.strip() and tid.strip() != "-1"
        ]

        row_parents = []
        row_labels = []

        for child_id in input_ids:
            best_ancestor = None
            best_distance = None

            for ancestor_id in ancestor_map.get(child_id, []):
                # Parent must occur somewhere in the combined
                # clinical + historical preclinical + update universe.
                if ancestor_id not in candidate_parent_ids:
                    continue

                depth = depth_to_root.get(
                    ancestor_id,
                    float("inf"),
                )

                if not (
                    min_depth
                    <= depth
                    < float("inf")
                ):
                    continue

                if desc_count.get(ancestor_id, 0) >= max_desc:
                    continue

                dist = distance_map.get(
                    (child_id, ancestor_id),
                    float("inf"),
                )

                if (
                    best_distance is None
                    or dist < best_distance
                ):
                    best_distance = dist
                    best_ancestor = ancestor_id

            if best_ancestor:
                row_parents.append(best_ancestor)

                try:
                    parent_label = ontology[best_ancestor].name
                except KeyError:
                    parent_label = best_ancestor

                row_labels.append(parent_label)

                if stats_dict is not None:
                    key = (
                        f"{parent_label}"
                        f"({best_ancestor})"
                    )
                    stats_dict.setdefault(
                        key,
                        [],
                    ).append(child_id)

            else:
                row_parents.append("-1")
                row_labels.append("-1")

        parent_ids.append(
            "|".join(row_parents)
            if row_parents
            else "-1"
        )

        parent_labels.append(
            "|".join(row_labels)
            if row_labels
            else "-1"
        )

    out = df.copy()

    out["nearest_dataset_parent_mondo"] = parent_ids
    out["nearest_dataset_parent_label"] = parent_labels

    return out


def merge_original_and_parent_mondo(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    *,
    parent_id_col: str = "nearest_dataset_parent_mondo",
    parent_label_col: str = "nearest_dataset_parent_label",
    out_id_col: str = "merged_mondo_termid",
    out_label_col: str = "merged_mondo_label",
    ignore_id: str = "-1",
    case_insensitive_labels: bool = True,
) -> pd.DataFrame:
    """
    Merge original MONDO terms with newly assigned dataset-parent terms.
    """
    merged_ids = []
    merged_labels = []

    for _, row in df.iterrows():
        orig_ids = str(
            row.get(id_col, "") or ""
        ).split("|")

        orig_labels = str(
            row.get(label_col, "") or ""
        ).split("|")

        parent_ids = str(
            row.get(parent_id_col, "") or ""
        ).split("|")

        parent_labels = str(
            row.get(parent_label_col, "") or ""
        ).split("|")

        mids = []
        mlabs = []
        seen = set()

        def _key(mid, mlab):
            mlab_norm = (
                mlab.strip().lower()
                if case_insensitive_labels
                else mlab.strip()
            )

            return (
                mid.strip(),
                mlab_norm,
            )

        # Keep original values first.
        for oid, olab in zip(
            orig_ids,
            orig_labels,
        ):
            oid = oid.strip()
            olab = olab.strip()

            if not oid and not olab:
                continue

            k = _key(oid, olab)

            if k not in seen:
                seen.add(k)
                mids.append(oid)
                mlabs.append(olab)

        # Add parents if not already represented.
        present_ids = {
            m.strip()
            for m in mids
            if m.strip()
        }

        for pid, plab in zip(
            parent_ids,
            parent_labels,
        ):
            pid = pid.strip()
            plab = plab.strip()

            if not pid or pid == ignore_id:
                continue

            if pid in present_ids:
                continue

            k = _key(pid, plab)

            if k not in seen:
                seen.add(k)
                present_ids.add(pid)
                mids.append(pid)
                mlabs.append(plab)

        merged_ids.append("|".join(mids))
        merged_labels.append("|".join(mlabs))

    out = df.copy()

    out[out_id_col] = merged_ids
    out[out_label_col] = merged_labels

    return out


# ------------------ main ------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assign MONDO dataset-parent terms to a new preclinical "
            "dataset using clinical, historical preclinical, and new "
            "preclinical terms as candidate parent nodes."
        )
    )

    parser.add_argument(
        "--clinical_reference_input",
        required=True,
        help=(
            "Historical clinical dataset used to contribute "
            "candidate MONDO parent terms."
        ),
    )

    parser.add_argument(
        "--preclinical_reference_input",
        required=True,
        help=(
            "Historical preclinical dataset used to contribute "
            "candidate MONDO parent terms."
        ),
    )

    parser.add_argument(
        "--preclinical_input",
        required=True,
        help=(
            "New preclinical dataset to map. Its MONDO terms also "
            "contribute to the candidate-parent universe."
        ),
    )

    parser.add_argument(
        "--preclinical_output",
        required=True,
    )

    parser.add_argument(
        "--ontology_path",
        required=True,
    )

    parser.add_argument(
        "--root_id",
        default="MONDO:0000001",
    )

    parser.add_argument(
        "--id_column",
        default="disease_mondo_termid",
    )

    parser.add_argument(
        "--label_column",
        default="disease_term_mondo_norm",
    )

    parser.add_argument(
        "--min_depth",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_desc",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--stats_output",
        default="./data/mondo/mondo_mapped_to_parents_preclinical_stats.csv",
    )

    args = parser.parse_args()

    print("Loading reference clinical data...")
    df_clinical_reference = pd.read_csv(
        args.clinical_reference_input,
        dtype=str,
    )

    print("Loading reference preclinical data...")
    df_preclinical_reference = pd.read_csv(
        args.preclinical_reference_input,
        dtype=str,
    )

    print("Loading new preclinical data...")
    df_preclinical = pd.read_csv(
        args.preclinical_input,
        dtype=str,
    )

    print("Loading MONDO ontology...")
    ontology = pronto.Ontology(
        args.ontology_path
    )

    # --------------------------------------------------
    # Build candidate-parent universe.
    # --------------------------------------------------

    ids_clinical = parse_mondo_ids(
        df_clinical_reference[
            args.id_column
        ]
    )

    ids_preclinical_reference = parse_mondo_ids(
        df_preclinical_reference[
            args.id_column
        ]
    )

    ids_preclinical_update = parse_mondo_ids(
        df_preclinical[
            args.id_column
        ]
    )

    candidate_parent_ids = (
        ids_clinical
        | ids_preclinical_reference
        | ids_preclinical_update
    )

    print(
        "Unique clinical MONDO IDs:",
        len(ids_clinical),
    )

    print(
        "Unique historical preclinical MONDO IDs:",
        len(ids_preclinical_reference),
    )

    print(
        "Unique update preclinical MONDO IDs:",
        len(ids_preclinical_update),
    )

    print(
        "Combined candidate parent IDs:",
        len(candidate_parent_ids),
    )

    # --------------------------------------------------
    # Build ontology maps.
    # --------------------------------------------------

    print("Building ancestor and distance maps...")

    ancestor_map, distance_map = (
        build_ancestor_and_distance_maps(
            ontology,
            candidate_parent_ids,
        )
    )

    candidates = set(
        ancestor_map.keys()
    )

    for ancestors in ancestor_map.values():
        candidates |= ancestors

    print(
        "Computing MONDO structural metrics for",
        len(candidates),
        "terms...",
    )

    depth_to_root, desc_count = (
        compute_mondo_term_metrics(
            ontology,
            args.root_id,
            candidates,
        )
    )

    # --------------------------------------------------
    # Map ONLY the new preclinical dataset.
    # --------------------------------------------------

    mapped_to_parent_preclinical = {}

    print(
        "Assigning parents to new preclinical dataset..."
    )

    df_p = assign_nearest_dataset_parents(
        df_preclinical,
        ontology,
        candidate_parent_ids,
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

    # --------------------------------------------------
    # Save only the new preclinical output.
    # --------------------------------------------------

    output_path = Path(
        args.preclinical_output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_p.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved mapped preclinical data to: {output_path}"
    )

    save_stats(
        mapped_to_parent_preclinical,
        args.stats_output,
    )

    print(
        f"Saved mapping statistics to: {args.stats_output}"
    )


if __name__ == "__main__":
    main()