#!/usr/bin/env python
"""Run GO/KEGG/Reactome over-representation analysis for dual-site gene groups."""

from __future__ import annotations

import argparse
import csv
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(rows: Sequence[Dict[str, object]], path: Path, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_gene(value: str) -> str:
    return value.strip().upper()


def truthy(value: object) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes", "Y"}


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    valid = [(idx, p) for idx, p in enumerate(p_values) if not math.isnan(p)]
    ranked = sorted(valid, key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running = 1.0
    total = len(ranked)
    for rank, (idx, p_value) in reversed(list(enumerate(ranked, start=1))):
        running = min(running, p_value * total / rank)
        adjusted[idx] = running
    return adjusted


def hypergeom_sf(k: int, population: int, successes: int, draws: int) -> float:
    try:
        from scipy.stats import hypergeom

        return float(hypergeom.sf(k - 1, population, successes, draws))
    except Exception:
        denom = math.comb(population, draws)
        p_value = 0.0
        for i in range(k, min(successes, draws) + 1):
            p_value += math.comb(successes, i) * math.comb(population - successes, draws - i) / denom
        return p_value


def run_ora(target: Set[str], background: Set[str], gene_sets: Dict[str, Set[str]], library_name: str) -> List[Dict[str, object]]:
    clean_background = {normalize_gene(g) for g in background if g}
    clean_target = {normalize_gene(g) for g in target if g} & clean_background
    rows = []
    for term, genes in gene_sets.items():
        term_genes = {normalize_gene(g) for g in genes if g} & clean_background
        if not term_genes:
            continue
        overlap = sorted(clean_target & term_genes)
        if not overlap:
            continue
        p_value = hypergeom_sf(len(overlap), len(clean_background), len(term_genes), len(clean_target))
        rows.append(
            {
                "library": library_name,
                "term": term,
                "overlap_count": len(overlap),
                "target_size": len(clean_target),
                "term_size_in_background": len(term_genes),
                "background_size": len(clean_background),
                "p_value": p_value,
                "overlap_genes": ";".join(overlap),
            }
        )
    rows.sort(key=lambda row: (float(row["p_value"]), -int(row["overlap_count"]), str(row["term"])))
    for row, q_value in zip(rows, bh_fdr([float(row["p_value"]) for row in rows])):
        row["fdr_bh"] = q_value
    return rows


def parse_gene_set_text(text: str) -> Dict[str, Set[str]]:
    gene_sets: Dict[str, Set[str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        term = parts[0]
        raw_genes = parts[2:] if len(parts) > 2 else parts[1:]
        genes: Set[str] = set()
        for chunk in raw_genes:
            for gene in chunk.replace(",", "\t").split("\t"):
                gene = normalize_gene(gene)
                if gene and not gene.startswith("HTTP"):
                    genes.add(gene)
        if genes:
            gene_sets[term] = genes
    return gene_sets


def fetch_enrichr_library(library_name: str) -> Dict[str, Set[str]]:
    query = urllib.parse.urlencode({"mode": "text", "libraryName": library_name})
    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    return parse_gene_set_text(text)


def load_gmt(path: Path) -> Dict[str, Set[str]]:
    return parse_gene_set_text(path.read_text(encoding="utf-8"))


def genes_for_group(rows: Sequence[Dict[str, str]], group: str) -> Set[str]:
    return {normalize_gene(row.get("gene_name", "")) for row in rows if truthy(row.get(group)) and row.get("gene_name")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GO/KEGG/Reactome ORA for current dual-site groups.")
    parser.add_argument("--membership", default="11_tables/biology_summary/transcript_group_membership.tsv")
    parser.add_argument("--out-dir", default="11_tables/biology_summary")
    parser.add_argument("--libraries", default="GO_Biological_Process_2023,KEGG_2021_Human,Reactome_2022")
    parser.add_argument("--gmt", action="append", default=[], help="Optional local GMT/text gene-set files.")
    parser.add_argument("--top-n", type=int, default=30)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    memberships = read_rows(Path(args.membership))
    background = {normalize_gene(row.get("gene_name", "")) for row in memberships if row.get("gene_name")}
    groups = [
        "m1A_3UTR_prox25",
        "m1A_3UTR_distal50",
        "m1A_CDS_prox25",
        "m1A_CDS_distal50",
        "m1A_exact_overlap_m6A",
        "m6A_only_same_transcript",
    ]
    libraries: Dict[str, Dict[str, Set[str]]] = {}
    for library in [name.strip() for name in args.libraries.split(",") if name.strip()]:
        libraries[library] = fetch_enrichr_library(library)
    for path_text in args.gmt:
        path = Path(path_text)
        libraries[path.stem] = load_gmt(path)

    all_rows = []
    for group in groups:
        target = genes_for_group(memberships, group)
        for library, gene_sets in libraries.items():
            for row in run_ora(target, background, gene_sets, library)[: args.top_n]:
                row["group"] = group
                all_rows.append(row)
    out_dir = Path(args.out_dir)
    write_table(
        [row for row in all_rows if row["library"].startswith("GO_")],
        out_dir / "go_enrichment.tsv",
        ["group", "library", "term", "overlap_count", "target_size", "term_size_in_background", "background_size", "p_value", "fdr_bh", "overlap_genes"],
    )
    write_table(
        [row for row in all_rows if not row["library"].startswith("GO_")],
        out_dir / "kegg_reactome_enrichment.tsv",
        ["group", "library", "term", "overlap_count", "target_size", "term_size_in_background", "background_size", "p_value", "fdr_bh", "overlap_genes"],
    )
    print(f"wrote {len(all_rows)} enrichment rows from {len(libraries)} libraries")


if __name__ == "__main__":
    main()
