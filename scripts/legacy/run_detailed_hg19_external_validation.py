from __future__ import annotations

import base64
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "11_tables" / "biology_summary"
SITE_GROUPS = BIO / "dual_site_groups.tsv"
MEMBERSHIP = BIO / "transcript_group_membership.tsv"
EXTERNAL = ROOT / "external" / "hg19_public" / "hg19_external_m1a_m6a_combined.tsv"
FIG_PNG = ROOT / "10_figures" / "biology_external_png"
FIG_SVG = ROOT / "10_figures" / "biology_external_svg"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

GRAY = ["#f9fafb", "#f3f4f6", "#e5e7eb", "#d1d5db", "#9ca3af", "#6b7280", "#4b5563", "#374151", "#111827"]
TEXT = "#111827"
MUTED = "#4b5563"
GRID = "#d1d5db"
PANEL = "#fbfbfc"

SPATIAL_GROUPS = [
    ("m1A_CDS_prox25", "CDS", "prox25", "CDS prox25"),
    ("m1A_CDS_intermediate26_50", "CDS", "intermediate26_50", "CDS 26-50"),
    ("m1A_CDS_distal50", "CDS", "distal50", "CDS distal"),
    ("m1A_3UTR_prox25", "3UTR", "prox25", "3UTR prox25"),
    ("m1A_3UTR_intermediate26_50", "3UTR", "intermediate26_50", "3UTR 26-50"),
    ("m1A_3UTR_distal50", "3UTR", "distal50", "3UTR distal"),
    ("m1A_other_region_prox25", "5UTR/other", "prox25", "5UTR/other prox25"),
    ("m1A_other_region_intermediate26_50", "5UTR/other", "intermediate26_50", "5UTR/other 26-50"),
    ("m1A_other_region_distal50", "5UTR/other", "distal50", "5UTR/other distal"),
]
CONTEXT_GROUPS = [
    ("m1A_exact_overlap_m6A", "exact m1A-m6A overlap"),
    ("m1A_no_m6A_same_transcript", "m1A without same-tx m6A"),
    ("m6A_only_same_transcript", "m6A only same-tx"),
    ("matched_unmodified_A", "matched unmodified A"),
]
GROUP_LABELS = {group: label for group, _, _, label in SPATIAL_GROUPS}
GROUP_LABELS.update(CONTEXT_GROUPS)
DISTANCE_ORDER = [
    ("prox25", "<=25 nt"),
    ("intermediate26_50", "26-50 nt"),
    ("distal50", ">50 nt"),
]
REGION_ORDER = ["CDS", "3UTR", "5UTR/other"]
STATUS_ORDER = [
    ("external_dual", "dual m1A+m6A", "#111827"),
    ("external_m1A_only", "m1A only", "#9ca3af"),
    ("external_m6A_only", "m6A only", "#4b5563"),
    ("external_none", "none", "#e5e7eb"),
]
EXTERNAL_REGION_COLORS = {
    "3UTR": "#111827",
    "CDS": "#6b7280",
    "5UTR": "#c7cbd1",
    "ncRNA": "#e5e7eb",
    "unknown": "#f3f4f6",
}
VIRTUAL_GROUPS = {
    "m1A_prox25_all": ["m1A_CDS_prox25", "m1A_3UTR_prox25", "m1A_other_region_prox25"],
    "m1A_intermediate26_50_all": [
        "m1A_CDS_intermediate26_50",
        "m1A_3UTR_intermediate26_50",
        "m1A_other_region_intermediate26_50",
    ],
    "m1A_distal50_all": ["m1A_CDS_distal50", "m1A_3UTR_distal50", "m1A_other_region_distal50"],
    "m1A_prox_or_intermediate_all": [
        "m1A_CDS_prox25",
        "m1A_3UTR_prox25",
        "m1A_other_region_prox25",
        "m1A_CDS_intermediate26_50",
        "m1A_3UTR_intermediate26_50",
        "m1A_other_region_intermediate26_50",
    ],
}
VIRTUAL_LABELS = {
    "m1A_prox25_all": "all m1A prox25",
    "m1A_intermediate26_50_all": "all m1A 26-50",
    "m1A_distal50_all": "all m1A distal",
    "m1A_prox_or_intermediate_all": "all m1A <=50 nt",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def normalize_symbol(value: str) -> str:
    return value.strip().upper()


def truthy(value: object) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes", "Y"}


def row_value(row: dict[str, str], candidates: Iterable[str]) -> str:
    lower_map = {key.lower(): key for key in row}
    for key in candidates:
        if key in row and row[key]:
            return str(row[key])
        mapped = lower_map.get(key.lower())
        if mapped and row.get(mapped):
            return str(row[mapped])
    return ""


def normalize_mod(value: str) -> str:
    clean = value.upper().replace("-", "")
    if "M1A" in clean:
        return "m1A"
    if "M6A" in clean:
        return "m6A"
    return ""


def load_external_summary() -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in read_tsv(EXTERNAL):
        mod = normalize_mod(row_value(row, ["mod_type", "RNAmodType", "modification"]))
        if mod not in {"m1A", "m6A"}:
            continue
        gene = normalize_symbol(row_value(row, ["gene_name", "geneName", "Gene", "gene"]))
        if not gene:
            continue
        current = summary.setdefault(
            gene,
            {
                "external_m1A_count": 0,
                "external_m6A_count": 0,
                "external_m1A_high_confidence_count": 0,
                "external_regions_m1A": Counter(),
                "external_regions_m6A": Counter(),
                "external_sources": Counter(),
                "external_evidence_tiers": Counter(),
                "chromosomes": set(),
                "transcripts": set(),
            },
        )
        current[f"external_{mod}_count"] = int(current[f"external_{mod}_count"]) + 1
        region = row_value(row, ["region_type", "Region", "region"]) or "unknown"
        current[f"external_regions_{mod}"][region] += 1
        source = row_value(row, ["source", "Source"]) or "unknown"
        current["external_sources"][source] += 1
        tier = row_value(row, ["evidence_tier", "evidence", "tier"]) or "not_reported"
        current["external_evidence_tiers"][tier] += 1
        if mod == "m1A" and "high_confidence" in tier:
            current["external_m1A_high_confidence_count"] = int(current["external_m1A_high_confidence_count"]) + 1
        chrom = row_value(row, ["chr", "chrom", "chromosome"])
        transcript = row_value(row, ["transcript_id", "transcriptID"])
        if chrom:
            current["chromosomes"].add(chrom)
        if transcript:
            current["transcripts"].add(transcript)
    return summary


def external_status(summary: dict[str, object] | None) -> str:
    if not summary:
        return "external_none"
    has_m1a = int(summary.get("external_m1A_count", 0)) > 0
    has_m6a = int(summary.get("external_m6A_count", 0)) > 0
    if has_m1a and has_m6a:
        return "external_dual"
    if has_m1a:
        return "external_m1A_only"
    if has_m6a:
        return "external_m6A_only"
    return "external_none"


def collapse_memberships(rows: list[dict[str, str]], key_field: str = "gene_name") -> list[dict[str, str]]:
    collapsed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = normalize_symbol(row.get(key_field, ""))
        if not key:
            continue
        out = collapsed.setdefault(key, {key_field: key})
        for col, value in row.items():
            if col == key_field:
                continue
            if truthy(value):
                out[col] = "1"
            elif col not in out:
                out[col] = value
    return list(collapsed.values())


def build_entity_sets(membership_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    collapsed = collapse_memberships(membership_rows)
    columns = [col for col in membership_rows[0].keys() if col not in {"transcript_id", "gene_id", "gene_name"}]
    sets: dict[str, set[str]] = {}
    for col in columns:
        sets[col] = {row["gene_name"] for row in collapsed if truthy(row.get(col))}
    for virtual, members in VIRTUAL_GROUPS.items():
        genes: set[str] = set()
        for member in members:
            genes.update(sets.get(member, set()))
        sets[virtual] = genes
    return sets


def fisher_exact(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    try:
        from scipy.stats import fisher_exact as scipy_fisher

        odds, p_value = scipy_fisher([[a, b], [c, d]])
        return float(odds), float(p_value)
    except Exception:
        n = a + b + c + d
        row1 = a + b
        col1 = a + c

        def log_choose(n_value: int, k_value: int) -> float:
            if k_value < 0 or k_value > n_value:
                return float("-inf")
            return math.lgamma(n_value + 1) - math.lgamma(k_value + 1) - math.lgamma(n_value - k_value + 1)

        def hyperprob(x: int) -> float:
            return math.exp(log_choose(col1, x) + log_choose(n - col1, row1 - x) - log_choose(n, row1))

        lo = max(0, row1 - (n - col1))
        hi = min(row1, col1)
        obs = hyperprob(a)
        p_value = sum(hyperprob(x) for x in range(lo, hi + 1) if hyperprob(x) <= obs + 1e-15)
        odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        return odds, min(1.0, p_value)


def bh_adjust(rows: list[dict[str, object]], key: str = "p_value") -> None:
    valid = [row for row in rows if not math.isnan(float(row[key]))]
    ranked = sorted(valid, key=lambda row: float(row[key]))
    total = len(ranked)
    running = 1.0
    for rank, row in reversed(list(enumerate(ranked, start=1))):
        running = min(running, float(row[key]) * total / rank)
        row["fdr_bh"] = running


def star(p_value: float) -> str:
    if math.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.05:
        return "**"
    return "ns"


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if math.isnan(value):
        return "#f3f4f6"
    if vmax <= vmin:
        return GRAY[4]
    t = max(0.0, min(0.999, (value - vmin) / (vmax - vmin)))
    return GRAY[int(t * len(GRAY))]


def fmt_fraction(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value * 100:.1f}%"


def counter_string(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in sorted(counter.items()))


def make_group_summary(site_rows: list[dict[str, str]], external_summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in site_rows:
        group_rows[row["site_group"]].append(row)

    meta = {group: ("spatial", region, distance, label) for group, region, distance, label in SPATIAL_GROUPS}
    for group, label in CONTEXT_GROUPS:
        meta[group] = ("context", "", "", label)

    rows: list[dict[str, object]] = []
    for group, rows_in_group in sorted(group_rows.items()):
        genes = {normalize_symbol(row.get("gene_name", "")) for row in rows_in_group if row.get("gene_name")}
        transcripts = {row.get("transcript_id", "") for row in rows_in_group if row.get("transcript_id")}
        statuses = Counter(external_status(external_summary.get(gene)) for gene in genes)
        m1a_genes = {gene for gene in genes if int(external_summary.get(gene, {}).get("external_m1A_count", 0)) > 0}
        m6a_genes = {gene for gene in genes if int(external_summary.get(gene, {}).get("external_m6A_count", 0)) > 0}
        dual_genes = {gene for gene in genes if external_status(external_summary.get(gene)) == "external_dual"}
        high_conf_m1a_genes = {
            gene for gene in genes if int(external_summary.get(gene, {}).get("external_m1A_high_confidence_count", 0)) > 0
        }
        supported_sites_dual = [row for row in rows_in_group if normalize_symbol(row.get("gene_name", "")) in dual_genes]
        supported_sites_any = [row for row in rows_in_group if normalize_symbol(row.get("gene_name", "")) in m1a_genes | m6a_genes]
        linked_m1a_counts = [int(external_summary[gene]["external_m1A_count"]) for gene in genes if gene in external_summary]
        linked_m6a_counts = [int(external_summary[gene]["external_m6A_count"]) for gene in genes if gene in external_summary]
        site_regions = Counter(row.get("region_type", "unknown") or "unknown" for row in rows_in_group)
        group_type, region, distance, label = meta.get(group, ("other", "", "", group))
        total = len(genes)
        rows.append(
            {
                "group": group,
                "label": label,
                "group_type": group_type,
                "region": region,
                "distance_class": distance,
                "site_count": len(rows_in_group),
                "transcript_count": len(transcripts),
                "gene_count": total,
                "external_any_gene_count": total - statuses["external_none"],
                "external_any_gene_fraction": (total - statuses["external_none"]) / total if total else 0.0,
                "external_m1A_gene_count": len(m1a_genes),
                "external_m1A_gene_fraction": len(m1a_genes) / total if total else 0.0,
                "external_m6A_gene_count": len(m6a_genes),
                "external_m6A_gene_fraction": len(m6a_genes) / total if total else 0.0,
                "external_dual_gene_count": len(dual_genes),
                "external_dual_gene_fraction": len(dual_genes) / total if total else 0.0,
                "external_m1A_only_gene_count": statuses["external_m1A_only"],
                "external_m6A_only_gene_count": statuses["external_m6A_only"],
                "external_none_gene_count": statuses["external_none"],
                "external_high_confidence_m1A_gene_count": len(high_conf_m1a_genes),
                "external_dual_supported_site_count_gene_level": len(supported_sites_dual),
                "external_dual_supported_site_fraction_gene_level": len(supported_sites_dual) / len(rows_in_group) if rows_in_group else 0.0,
                "external_any_supported_site_count_gene_level": len(supported_sites_any),
                "external_any_supported_site_fraction_gene_level": len(supported_sites_any) / len(rows_in_group) if rows_in_group else 0.0,
                "external_m1A_annotation_count_linked_genes": sum(linked_m1a_counts),
                "external_m6A_annotation_count_linked_genes": sum(linked_m6a_counts),
                "median_external_m1A_count_annotated_gene": median([x for x in linked_m1a_counts if x > 0]) if any(x > 0 for x in linked_m1a_counts) else 0,
                "median_external_m6A_count_annotated_gene": median([x for x in linked_m6a_counts if x > 0]) if any(x > 0 for x in linked_m6a_counts) else 0,
                "site_region_breakdown": counter_string(site_regions),
            }
        )
    return rows


def make_region_distance_summary(group_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = {group for group, _, _, _ in SPATIAL_GROUPS}
    rows = [row for row in group_summary if row["group"] in wanted]
    rows.sort(key=lambda row: (REGION_ORDER.index(str(row["region"])), [d for d, _ in DISTANCE_ORDER].index(str(row["distance_class"]))))
    return rows


def comparison_pairs() -> list[tuple[str, str, str, str]]:
    return [
        ("m1A_3UTR_prox25", "m1A_3UTR_distal50", "3UTR prox25 vs distal", "regional_distance"),
        ("m1A_3UTR_intermediate26_50", "m1A_3UTR_distal50", "3UTR 26-50 vs distal", "regional_distance"),
        ("m1A_CDS_prox25", "m1A_CDS_distal50", "CDS prox25 vs distal", "regional_distance"),
        ("m1A_CDS_intermediate26_50", "m1A_CDS_distal50", "CDS 26-50 vs distal", "regional_distance"),
        ("m1A_other_region_prox25", "m1A_other_region_distal50", "5UTR/other prox25 vs distal", "regional_distance"),
        ("m1A_other_region_intermediate26_50", "m1A_other_region_distal50", "5UTR/other 26-50 vs distal", "regional_distance"),
        ("m1A_prox25_all", "m1A_distal50_all", "all m1A prox25 vs distal", "pooled_distance"),
        ("m1A_prox_or_intermediate_all", "m1A_distal50_all", "all m1A <=50 nt vs distal", "pooled_distance"),
        ("m1A_exact_overlap_m6A", "m1A_no_m6A_same_transcript", "exact overlap vs m1A no same-tx m6A", "dual_context"),
        ("m1A_exact_overlap_m6A", "m6A_only_same_transcript", "exact overlap vs m6A only same-tx", "dual_context"),
        ("m1A_3UTR_prox25", "m6A_only_same_transcript", "3UTR prox25 vs m6A only same-tx", "cross_context"),
        ("m1A_CDS_prox25", "m6A_only_same_transcript", "CDS prox25 vs m6A only same-tx", "cross_context"),
        ("m1A_no_m6A_same_transcript", "m6A_only_same_transcript", "m1A no same-tx m6A vs m6A only", "cross_context"),
    ]


def outcome_sets(external_summary: dict[str, dict[str, object]]) -> dict[str, set[str]]:
    return {
        "external_any": {gene for gene, value in external_summary.items() if external_status(value) != "external_none"},
        "external_m1A_any": {gene for gene, value in external_summary.items() if int(value.get("external_m1A_count", 0)) > 0},
        "external_m6A_any": {gene for gene, value in external_summary.items() if int(value.get("external_m6A_count", 0)) > 0},
        "external_dual": {gene for gene, value in external_summary.items() if external_status(value) == "external_dual"},
    }


def make_enrichment_rows(entity_sets: dict[str, set[str]], external_summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    outcomes = outcome_sets(external_summary)
    labels = dict(GROUP_LABELS)
    labels.update(VIRTUAL_LABELS)
    rows: list[dict[str, object]] = []
    for outcome, positive_genes in outcomes.items():
        for left, right, comparison, comparison_type in comparison_pairs():
            left_entities = set(entity_sets.get(left, set()))
            right_entities = set(entity_sets.get(right, set())) - left_entities
            left_pos = len(left_entities & positive_genes)
            right_pos = len(right_entities & positive_genes)
            left_total = len(left_entities)
            right_total = len(right_entities)
            odds, p_value = fisher_exact(left_pos, left_total - left_pos, right_pos, right_total - right_pos)
            left_fraction = left_pos / left_total if left_total else 0.0
            right_fraction = right_pos / right_total if right_total else 0.0
            rows.append(
                {
                    "outcome": outcome,
                    "comparison": comparison,
                    "comparison_type": comparison_type,
                    "left_group": left,
                    "right_group": right,
                    "left_label": labels.get(left, left),
                    "right_label": labels.get(right, right),
                    "left_total_genes": left_total,
                    "right_total_genes_nonoverlap": right_total,
                    "left_external_positive_genes": left_pos,
                    "right_external_positive_genes": right_pos,
                    "left_external_fraction": left_fraction,
                    "right_external_fraction": right_fraction,
                    "risk_difference": left_fraction - right_fraction,
                    "odds_ratio": odds,
                    "log2_odds_ratio": math.log2(odds) if odds > 0 and not math.isinf(odds) else float("nan"),
                    "p_value": p_value,
                    "fdr_bh": 1.0,
                    "significance": star(p_value),
                }
            )
    bh_adjust(rows)
    for row in rows:
        row["significance"] = star(float(row["fdr_bh"]))
    return rows


def make_status_rows(group_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_row in group_summary:
        total = int(group_row["gene_count"])
        for status, label, _ in STATUS_ORDER:
            count_key = {
                "external_dual": "external_dual_gene_count",
                "external_m1A_only": "external_m1A_only_gene_count",
                "external_m6A_only": "external_m6A_only_gene_count",
                "external_none": "external_none_gene_count",
            }[status]
            count = int(group_row[count_key])
            rows.append(
                {
                    "group": group_row["group"],
                    "label": group_row["label"],
                    "status": status,
                    "status_label": label,
                    "count": count,
                    "total_genes": total,
                    "fraction": count / total if total else 0.0,
                }
            )
    return rows


def make_region_by_group_rows(
    site_rows: list[dict[str, str]],
    group_summary: list[dict[str, object]],
    external_summary: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    groups = {str(row["group"]) for row in group_summary}
    genes_by_group: dict[str, set[str]] = defaultdict(set)
    for row in site_rows:
        group = row["site_group"]
        if group not in groups:
            continue
        gene = normalize_symbol(row.get("gene_name", ""))
        if gene:
            genes_by_group[group].add(gene)

    rows: list[dict[str, object]] = []
    for group, genes in sorted(genes_by_group.items()):
        for mod in ["m1A", "m6A"]:
            totals: Counter[str] = Counter()
            for gene in genes:
                summary = external_summary.get(gene)
                if not summary:
                    continue
                totals.update(summary[f"external_regions_{mod}"])
            total_count = sum(totals.values())
            for region in sorted(totals):
                rows.append(
                    {
                        "group": group,
                        "label": GROUP_LABELS.get(group, group),
                        "mod_type": mod,
                        "external_region": region,
                        "external_annotation_count": totals[region],
                        "total_linked_external_annotations": total_count,
                        "fraction": totals[region] / total_count if total_count else 0.0,
                    }
                )
    return rows


def make_gene_overlap_rows(
    site_rows: list[dict[str, str]],
    external_summary: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    groups_by_gene: dict[str, set[str]] = defaultdict(set)
    sites_by_gene: Counter[str] = Counter()
    transcripts_by_gene: dict[str, set[str]] = defaultdict(set)
    for row in site_rows:
        gene = normalize_symbol(row.get("gene_name", ""))
        if not gene:
            continue
        groups_by_gene[gene].add(row["site_group"])
        sites_by_gene[gene] += 1
        transcript = row.get("transcript_id", "")
        if transcript:
            transcripts_by_gene[gene].add(transcript)

    rows: list[dict[str, object]] = []
    for gene, summary in sorted(external_summary.items()):
        local_groups = groups_by_gene.get(gene, set())
        if not local_groups:
            continue
        rows.append(
            {
                "gene_name": gene,
                "external_status": external_status(summary),
                "external_m1A_count": summary["external_m1A_count"],
                "external_m6A_count": summary["external_m6A_count"],
                "external_m1A_high_confidence_count": summary["external_m1A_high_confidence_count"],
                "external_m1A_regions": counter_string(summary["external_regions_m1A"]),
                "external_m6A_regions": counter_string(summary["external_regions_m6A"]),
                "local_site_count": sites_by_gene[gene],
                "local_transcript_count": len(transcripts_by_gene[gene]),
                "positive_group_count": len(local_groups),
                "positive_groups": ";".join(sorted(local_groups)),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["external_status"]) != "external_dual",
            -int(row["external_m1A_count"]) - int(row["external_m6A_count"]),
            -int(row["positive_group_count"]),
            str(row["gene_name"]),
        )
    )
    return rows


def make_source_summary_rows(external_summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    source_counter: Counter[str] = Counter()
    tier_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    for summary in external_summary.values():
        source_counter.update(summary["external_sources"])
        tier_counter.update(summary["external_evidence_tiers"])
        status_counter[external_status(summary)] += 1

    rows: list[dict[str, object]] = []
    for key, count in sorted(source_counter.items()):
        rows.append({"summary_type": "source_annotation_rows", "category": key, "count": count})
    for key, count in sorted(tier_counter.items()):
        rows.append({"summary_type": "evidence_tier_annotation_rows", "category": key, "count": count})
    for key, count in sorted(status_counter.items()):
        rows.append({"summary_type": "gene_external_status", "category": key, "count": count})
    return rows


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=12, fill=PANEL, outline=GRID, width=2)
    x1, y1, _, _ = box
    draw.text((x1 + 24, y1 + 16), title, font=font(23, True), fill=TEXT)


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, fnt: ImageFont.ImageFont, fill: str) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_size(draw, candidate, fnt)[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += 23


def make_heatmap(draw: ImageDraw.ImageDraw, region_rows: list[dict[str, object]]) -> None:
    x0, y0, w, h = 48, 120, 690, 430
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "A  External dual-positive fraction")
    by_key = {(str(row["region"]), str(row["distance_class"])): row for row in region_rows}
    cell_w, cell_h = 162, 84
    bx, by = x0 + 160, y0 + 100
    values = [float(row["external_dual_gene_fraction"]) for row in region_rows]
    vmin, vmax = min(values), max(values)
    for c, (distance, label) in enumerate(DISTANCE_ORDER):
        draw.text((bx + c * cell_w + 20, by - 38), label, font=font(15, True), fill=MUTED)
    for r, region in enumerate(REGION_ORDER):
        draw.text((x0 + 30, by + r * cell_h + 28), region, font=font(17, True), fill=TEXT)
        for c, (distance, _) in enumerate(DISTANCE_ORDER):
            row = by_key.get((region, distance))
            value = float(row["external_dual_gene_fraction"]) if row else float("nan")
            x = bx + c * cell_w
            y = by + r * cell_h
            fill = color_scale(value, vmin, vmax)
            text_fill = "#ffffff" if value > (vmin + vmax) / 2 else TEXT
            draw.rounded_rectangle([x, y, x + cell_w - 12, y + cell_h - 12], radius=8, fill=fill, outline="#9ca3af", width=1)
            if row:
                draw.text((x + 31, y + 18), fmt_fraction(value), font=font(22, True), fill=text_fill)
                draw.text((x + 22, y + 51), f"{row['external_dual_gene_count']}/{row['gene_count']} genes", font=font(12), fill=text_fill)
    draw.text((x0 + 30, y0 + h - 42), "Dual-positive means external m1A and external m6A annotations co-occur at gene level.", font=font(13), fill=MUTED)


def make_or_panel(draw: ImageDraw.ImageDraw, enrichment_rows: list[dict[str, object]]) -> None:
    x0, y0, w, h = 770, 120, 982, 590
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "B  External dual-positive enrichment contrasts")
    selected_names = [
        "3UTR prox25 vs distal",
        "3UTR 26-50 vs distal",
        "CDS prox25 vs distal",
        "CDS 26-50 vs distal",
        "5UTR/other prox25 vs distal",
        "all m1A prox25 vs distal",
        "all m1A <=50 nt vs distal",
        "exact overlap vs m1A no same-tx m6A",
        "3UTR prox25 vs m6A only same-tx",
        "CDS prox25 vs m6A only same-tx",
    ]
    rows_by_name = {str(row["comparison"]): row for row in enrichment_rows if row["outcome"] == "external_dual"}
    rows = [rows_by_name[name] for name in selected_names if name in rows_by_name]
    plot_x, plot_y = x0 + 352, y0 + 78
    plot_w, plot_h = 520, 430
    min_log, max_log = -0.55, 1.8

    def sx(log_value: float) -> int:
        return plot_x + int((max(min_log, min(max_log, log_value)) - min_log) / (max_log - min_log) * plot_w)

    for tick in [-0.5, 0, 0.5, 1.0, 1.5]:
        tx = sx(tick)
        draw.line([tx, plot_y - 12, tx, plot_y + plot_h], fill="#e5e7eb", width=1)
        label = "1" if tick == 0 else f"{2 ** tick:.1f}"
        draw.text((tx - 12, plot_y + plot_h + 10), label, font=font(12), fill=MUTED)
    draw.line([sx(0), plot_y - 18, sx(0), plot_y + plot_h], fill="#111827", width=2)
    for idx, row in enumerate(rows):
        y = plot_y + 18 + idx * 40
        odds = float(row["odds_ratio"])
        log_value = math.log2(odds) if odds > 0 and not math.isinf(odds) else max_log
        x = sx(log_value)
        draw.text((x0 + 26, y - 10), str(row["comparison"])[:43], font=font(13), fill=TEXT)
        draw.line([sx(0), y, x, y], fill="#6b7280", width=8)
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill="#111827", outline="#000000", width=1)
        draw.text((plot_x + plot_w + 16, y - 12), f"OR {odds:.2f} {row['significance']}", font=font(13, True), fill=TEXT)
    draw.text((plot_x + 190, y0 + h - 42), "odds ratio, log2-scaled axis", font=font(14, True), fill=MUTED)
    draw.text((x0 + 26, y0 + h - 42), "*** FDR < 0.001; ** FDR < 0.05", font=font(13), fill=MUTED)


def make_status_panel(draw: ImageDraw.ImageDraw, status_rows: list[dict[str, object]]) -> None:
    x0, y0, w, h = 48, 590, 690, 545
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "C  External annotation status by local group")
    selected = [
        "m1A_exact_overlap_m6A",
        "m1A_3UTR_prox25",
        "m1A_CDS_prox25",
        "m1A_other_region_prox25",
        "m1A_no_m6A_same_transcript",
        "m6A_only_same_transcript",
        "matched_unmodified_A",
    ]
    by_group: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in status_rows:
        by_group[str(row["group"])][str(row["status"])] = row
    bar_x, bar_y, bar_w = x0 + 245, y0 + 92, 360
    for idx, group in enumerate(selected):
        status_map = by_group.get(group, {})
        if not status_map:
            continue
        y = bar_y + idx * 55
        draw.text((x0 + 26, y - 4), str(status_map["external_dual"]["label"])[:28], font=font(13), fill=TEXT)
        x = bar_x
        for status, _, color in STATUS_ORDER:
            frac = float(status_map.get(status, {}).get("fraction", 0))
            seg_w = int(bar_w * frac)
            draw.rectangle([x, y, x + seg_w, y + 26], fill=color, outline="#ffffff")
            x += seg_w
        draw.rectangle([bar_x, y, bar_x + bar_w, y + 26], outline="#9ca3af", width=1)
        dual_frac = float(status_map["external_dual"]["fraction"])
        any_frac = 1.0 - float(status_map["external_none"]["fraction"])
        draw.text((bar_x + bar_w + 12, y - 1), f"dual {dual_frac*100:.1f}%", font=font(12, True), fill=TEXT)
        draw.text((bar_x + bar_w + 12, y + 18), f"any {any_frac*100:.1f}%", font=font(11), fill=MUTED)
    lx, ly = x0 + 28, y0 + h - 72
    for idx, (_, label, color) in enumerate(STATUS_ORDER):
        x = lx + idx * 158
        draw.rectangle([x, ly, x + 16, ly + 16], fill=color, outline="#9ca3af")
        draw.text((x + 22, ly - 2), label, font=font(11), fill=TEXT)


def make_region_panel(draw: ImageDraw.ImageDraw, region_rows: list[dict[str, object]]) -> None:
    x0, y0, w, h = 770, 750, 982, 385
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "D  Linked external annotation region composition")
    selected = ["m1A_3UTR_prox25", "m1A_CDS_prox25", "m1A_exact_overlap_m6A", "m6A_only_same_transcript"]
    by_group_mod: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in region_rows:
        by_group_mod[(str(row["group"]), str(row["mod_type"]))].append(row)
    bar_x, bar_y, bar_w = x0 + 285, y0 + 70, 520
    lane_h = 18
    row_step = 68
    for idx, group in enumerate(selected):
        y_base = bar_y + idx * row_step
        draw.text((x0 + 26, y_base + 15), GROUP_LABELS.get(group, group)[:30], font=font(13), fill=TEXT)
        for lane, mod in enumerate(["m1A", "m6A"]):
            rows = by_group_mod.get((group, mod), [])
            y = y_base + lane * 26
            draw.text((bar_x - 44, y - 1), mod, font=font(12, True), fill=MUTED)
            x = bar_x
            total = sum(int(row["external_annotation_count"]) for row in rows)
            for row in sorted(rows, key=lambda item: str(item["external_region"])):
                region = str(row["external_region"])
                frac = float(row["fraction"])
                seg_w = int(bar_w * frac)
                color = EXTERNAL_REGION_COLORS.get(region, "#9ca3af")
                draw.rectangle([x, y, x + seg_w, y + lane_h], fill=color, outline="#ffffff")
                x += seg_w
            draw.rectangle([bar_x, y, bar_x + bar_w, y + lane_h], outline="#9ca3af", width=1)
            draw.text((bar_x + bar_w + 12, y - 3), f"{total}", font=font(11), fill=MUTED)
    lx, ly = x0 + 28, y0 + h - 38
    for idx, region in enumerate(["3UTR", "CDS", "5UTR", "ncRNA"]):
        x = lx + idx * 130
        draw.rectangle([x, ly, x + 16, ly + 16], fill=EXTERNAL_REGION_COLORS.get(region, "#9ca3af"), outline="#9ca3af")
        draw.text((x + 22, ly - 2), region, font=font(12), fill=TEXT)


def make_note_panel(draw: ImageDraw.ImageDraw) -> None:
    x0, y0, w, h = 48, 1172, 1704, 158
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=12, fill="#ffffff", outline=GRID, width=2)
    draw.text((x0 + 26, y0 + 22), "Interpretation boundary", font=font(22, True), fill=TEXT)
    note = (
        "External validation uses independent hg19-compatible public m1A-miCLIP and m6A peak annotations aggregated by gene symbol. "
        "Dual-positive support means that a local group gene has at least one external m1A annotation and at least one external m6A annotation. "
        "This tests concordance with public modification annotations, not prospective held-out model performance or base-resolution validation."
    )
    draw_wrapped_text(draw, (x0 + 26, y0 + 60), note, 1600, font(16), MUTED)


def make_png(
    group_summary: list[dict[str, object]],
    region_distance: list[dict[str, object]],
    enrichment: list[dict[str, object]],
    status_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    out: Path,
) -> None:
    width, height = 1800, 1380
    canvas = Image.new("RGBA", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 30), "Detailed hg19 external m1A/m6A annotation validation", font=font(34, True), fill=TEXT)
    draw.text(
        (48, 74),
        "Grey palette; independent public m1A-miCLIP and m6A peak annotations are compared with local StructDualRM spatial groups.",
        font=font(18),
        fill=MUTED,
    )
    make_heatmap(draw, region_distance)
    make_or_panel(draw, enrichment)
    make_status_panel(draw, status_rows)
    make_region_panel(draw, region_rows)
    make_note_panel(draw)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


def make_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1380" viewBox="0 0 1800 1380">\n'
        f'<image href="data:image/png;base64,{data}" x="0" y="0" width="1800" height="1380"/>\n'
        "</svg>\n"
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8", newline="\n")


def update_figure_manifest(entries: list[tuple[str, str, Path, Path]]) -> None:
    manifest = ROOT / "10_figures" / "figure_manifest.tsv"
    fields = ["figure", "format", "source_path", "project_path", "size_bytes"]
    existing: list[dict[str, str]] = []
    if manifest.exists():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
    remove = {(figure, fmt) for figure, fmt, _, _ in entries}
    kept = [row for row in existing if (row.get("figure", ""), row.get("format", "")) not in remove]
    for figure, fmt, submission_path, project_path in entries:
        kept.append(
            {
                "figure": figure,
                "format": fmt,
                "source_path": str(submission_path.relative_to(ROOT)),
                "project_path": str(project_path.relative_to(ROOT)),
                "size_bytes": project_path.stat().st_size if project_path.exists() else 0,
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    site_rows = read_tsv(SITE_GROUPS)
    membership_rows = read_tsv(MEMBERSHIP)
    external_summary = load_external_summary()
    entity_sets = build_entity_sets(membership_rows)

    group_summary = make_group_summary(site_rows, external_summary)
    region_distance = make_region_distance_summary(group_summary)
    enrichment = make_enrichment_rows(entity_sets, external_summary)
    status_rows = make_status_rows(group_summary)
    region_by_group = make_region_by_group_rows(site_rows, group_summary, external_summary)
    gene_overlap = make_gene_overlap_rows(site_rows, external_summary)
    source_summary = make_source_summary_rows(external_summary)

    group_fields = [
        "group",
        "label",
        "group_type",
        "region",
        "distance_class",
        "site_count",
        "transcript_count",
        "gene_count",
        "external_any_gene_count",
        "external_any_gene_fraction",
        "external_m1A_gene_count",
        "external_m1A_gene_fraction",
        "external_m6A_gene_count",
        "external_m6A_gene_fraction",
        "external_dual_gene_count",
        "external_dual_gene_fraction",
        "external_m1A_only_gene_count",
        "external_m6A_only_gene_count",
        "external_none_gene_count",
        "external_high_confidence_m1A_gene_count",
        "external_dual_supported_site_count_gene_level",
        "external_dual_supported_site_fraction_gene_level",
        "external_any_supported_site_count_gene_level",
        "external_any_supported_site_fraction_gene_level",
        "external_m1A_annotation_count_linked_genes",
        "external_m6A_annotation_count_linked_genes",
        "median_external_m1A_count_annotated_gene",
        "median_external_m6A_count_annotated_gene",
        "site_region_breakdown",
    ]
    write_tsv(BIO / "hg19_external_group_summary.tsv", group_summary, group_fields)
    write_tsv(BIO / "hg19_external_region_distance_summary.tsv", region_distance, group_fields)
    write_tsv(
        BIO / "hg19_external_detailed_enrichment.tsv",
        enrichment,
        [
            "outcome",
            "comparison",
            "comparison_type",
            "left_group",
            "right_group",
            "left_label",
            "right_label",
            "left_total_genes",
            "right_total_genes_nonoverlap",
            "left_external_positive_genes",
            "right_external_positive_genes",
            "left_external_fraction",
            "right_external_fraction",
            "risk_difference",
            "odds_ratio",
            "log2_odds_ratio",
            "p_value",
            "fdr_bh",
            "significance",
        ],
    )
    write_tsv(BIO / "hg19_external_status_by_group.tsv", status_rows, ["group", "label", "status", "status_label", "count", "total_genes", "fraction"])
    write_tsv(
        BIO / "hg19_external_annotation_region_by_group.tsv",
        region_by_group,
        ["group", "label", "mod_type", "external_region", "external_annotation_count", "total_linked_external_annotations", "fraction"],
    )
    write_tsv(
        BIO / "hg19_external_gene_overlap_summary.tsv",
        gene_overlap,
        [
            "gene_name",
            "external_status",
            "external_m1A_count",
            "external_m6A_count",
            "external_m1A_high_confidence_count",
            "external_m1A_regions",
            "external_m6A_regions",
            "local_site_count",
            "local_transcript_count",
            "positive_group_count",
            "positive_groups",
        ],
    )
    write_tsv(BIO / "hg19_external_annotation_source_summary.tsv", source_summary, ["summary_type", "category", "count"])

    png = FIG_PNG / "Figure_hg19_external_validation_detail_gray.png"
    svg = FIG_SVG / "Figure_hg19_external_validation_detail_gray.svg"
    make_png(group_summary, region_distance, enrichment, status_rows, region_by_group, png)
    make_svg_wrapper(png, svg)

    copies = [
        (png, PKG / "figures_png" / png.name),
        (svg, PKG / "figures" / svg.name),
        (BIO / "hg19_external_group_summary.tsv", PKG / "source_data" / "Figure_hg19_external_group_summary.tsv"),
        (BIO / "hg19_external_region_distance_summary.tsv", PKG / "source_data" / "Figure_hg19_external_region_distance_summary.tsv"),
        (BIO / "hg19_external_detailed_enrichment.tsv", PKG / "source_data" / "Figure_hg19_external_detailed_enrichment.tsv"),
        (BIO / "hg19_external_status_by_group.tsv", PKG / "source_data" / "Figure_hg19_external_status_by_group.tsv"),
        (BIO / "hg19_external_annotation_region_by_group.tsv", PKG / "source_data" / "Figure_hg19_external_annotation_region_by_group.tsv"),
        (BIO / "hg19_external_gene_overlap_summary.tsv", PKG / "source_data" / "Figure_hg19_external_gene_overlap_summary.tsv"),
        (BIO / "hg19_external_annotation_source_summary.tsv", PKG / "source_data" / "Figure_hg19_external_annotation_source_summary.tsv"),
    ]
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    update_figure_manifest(
        [
            ("Figure_hg19_external_validation_detail_gray", "svg", PKG / "figures" / svg.name, svg),
            ("Figure_hg19_external_validation_detail_gray", "png", PKG / "figures_png" / png.name, png),
        ]
    )
    print(
        {
            "external_annotated_genes": len(external_summary),
            "groups": len(group_summary),
            "enrichment_rows": len(enrichment),
            "gene_overlap_rows": len(gene_overlap),
            "png": str(png),
        }
    )


if __name__ == "__main__":
    main()
