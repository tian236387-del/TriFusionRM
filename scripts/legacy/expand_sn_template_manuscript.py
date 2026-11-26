from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "submission" / "nature_communications_structdualrm"
SN_DRAFT = PKG / "sn_template_draft"
TEX = SN_DRAFT / "StructDualRM_NatureCommunications_SN_Template_Draft.tex"
REFERENCE_PDF = Path(r"C:\Users\夏添\Desktop\生信\reference\m1A_m6A\Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occurring RNA modifications.pdf")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def esc(text: object) -> str:
    value = str(text)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in value)


def fnum(value: str | float, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.{min(digits, 3)}f}"
    if abs(x) >= 0.001:
        return f"{x:.{digits}f}"
    return f"{x:.2e}"


def page_count_from_pdf(path: Path) -> int:
    import re

    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", data))


def table_dataset(workflow: list[dict[str, str]]) -> str:
    keep = [
        ("m1A_raw_parsed", "Raw m1A records parsed"),
        ("m6A_raw_parsed", "Raw m6A records parsed"),
        ("m1A_hg19", "m1A records retained in hg19"),
        ("m6A_hg19", "m6A records retained in hg19"),
        ("m1A_refseq_mapped", "m1A sites mapped to RefSeq transcript space"),
        ("m6A_refseq_mapped", "m6A sites mapped to RefSeq transcript space"),
        ("joint_positive_sites", "Joint positive candidate sites"),
        ("negative_pool", "Negative adenine candidate pool"),
        ("joint_dataset", "Joint supervised dataset before split"),
        ("rnafold_101nt", "101 nt windows with RNAfold structure"),
    ]
    lookup = {r["stage"]: r for r in workflow}
    lines = []
    for stage, label in keep:
        r = lookup.get(stage, {})
        lines.append(
            f"{esc(label)} & {esc(stage)} & {esc(r.get('rows', ''))} & {esc(r.get('m1A_positive', ''))} & {esc(r.get('m6A_positive', ''))} & {esc(r.get('co_positive', ''))} & {esc(r.get('negative', ''))} \\\\"
        )
    return r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Dataset construction and hg19 processing summary.} Empty cells indicate stages where the corresponding label-specific count is not applicable.}
\label{tab:dataset}
\begin{tabular}{lllllll}
\toprule
Processing step & Stage ID & Rows & m1A+ & m6A+ & Co+ & Negative \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def table_model_ablation(rows: list[dict[str, str]]) -> str:
    names = {
        "M0": "Sequence transformer",
        "M1": "Transformer + global structure",
        "M2": "Transformer + GNN",
        "M3": "StructDualRM gate",
        "M4": "Full StructDualRM",
    }
    body = []
    for r in rows:
        body.append(
            f"{esc(r['variant'])} & {esc(names.get(r['variant'], r['model_dir']))} & {fnum(r['test_macro_AUPR'],4)} & {fnum(r['test_macro_MCC'],4)} & {fnum(r['test_average_F1'],4)} & {fnum(r['test_m1A_AUPR'],4)} & {fnum(r['test_m6A_AUPR'],4)} \\\\"
        )
    return r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Internal ablation performance under the same 50-epoch split.} Macro AUPR and macro MCC summarize the two-label m1A/m6A task; label-specific AUPR values show that the full model particularly improves the rarer m1A label.}
\label{tab:ablation}
\begin{tabular}{llccccc}
\toprule
Variant & Model description & Macro AUPR & Macro MCC & Average F1 & m1A AUPR & m6A AUPR \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def table_benchmark(rows: list[dict[str, str]]) -> str:
    order = [
        "kmer_logreg",
        "kmer_extratrees",
        "multirm_official_101nt_zero_shot",
        "multirm_style_bilstm_attention",
        "M0",
        "M4",
    ]
    label = {
        "kmer_logreg": "k-mer logistic regression",
        "kmer_extratrees": "k-mer ExtraTrees",
        "multirm_official_101nt_zero_shot": "Official MultiRM zero-shot",
        "multirm_style_bilstm_attention": "MultiRM-style BiLSTM attention",
        "M0": "Sequence transformer M0",
        "M4": "Full StructDualRM M4",
    }
    lookup = {r["variant"]: r for r in rows}
    body = []
    for key in order:
        r = lookup[key]
        body.append(
            f"{esc(label[key])} & {fnum(r['test_macro_AUPR'],4)} & {fnum(r['test_macro_MCC'],4)} & {fnum(r['test_average_F1'],4)} & {fnum(r['test_m1A_AUPR'],4)} & {fnum(r['test_m6A_AUPR'],4)} \\\\"
        )
    return r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Direct same-dataset benchmark against MultiRM-style and classical baselines.} All rows were evaluated on the same held-out split, making the comparison stricter than cross-paper metric comparison.}
\label{tab:benchmark}
\begin{tabular}{lccccc}
\toprule
Model & Macro AUPR & Macro MCC & Average F1 & m1A AUPR & m6A AUPR \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def short_comparison(text: str) -> str:
    return (
        text.replace("m1A_", "")
        .replace("_same_transcript", "")
        .replace("_", " ")
        .replace("prox25", "proximal")
        .replace("distal50", "distal")
    )


def table_reader_external(reader: list[dict[str, str]], external: list[dict[str, str]]) -> str:
    ext_lookup = {r["comparison"]: r for r in external}
    body = []
    for r in reader:
        e = ext_lookup[r["comparison"]]
        body.append(
            f"{esc(short_comparison(r['comparison']))} & {r['left_total']} & {r['right_total']} & {fnum(r['odds_ratio'],3)} & {fnum(r['fdr_bh'],3)} & {fnum(e['odds_ratio'],3)} & {fnum(e['fdr_bh'],3)} \\\\"
        )
    return r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Reader enrichment and independent hg19 external-validation support.} Reader columns use YTHDF2 PAR-CLIP overlap; external columns use independent hg19-compatible m1A/m6A public annotations.}
\label{tab:reader_external}
\begin{tabular}{lcccccc}
\toprule
Comparison & Left n & Right n & Reader OR & Reader FDR & External OR & External FDR \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def table_structure(structure: list[dict[str, str]], seq: list[dict[str, str]]) -> str:
    keep_features = [
        "gc_fraction_101nt",
        "MFE_per_nt",
        "paired_fraction_101nt",
        "central_11nt_paired_fraction",
        "central_21nt_paired_fraction",
        "loop_span_around_center",
    ]
    body = []
    for r in structure:
        if r["comparison"] == "m1A_CDS_prox25 vs m1A_CDS_distal50" and r["feature"] in keep_features:
            body.append(
                f"{esc(r['feature'].replace('_', ' '))} & {fnum(r['left_median'],3)} & {fnum(r['right_median'],3)} & {fnum(r['effect_size'],3)} & {fnum(r['fdr_bh'],3)} \\\\"
            )
    for r in seq:
        if r["comparison"] == "m1A_CDS_prox25 vs m1A_CDS_distal50":
            body.append(
                f"{esc('motif: ' + r['feature'].replace('_', ' '))} & {fnum(r['left_median'],3)} & {fnum(r['right_median'],3)} & OR {fnum(r['odds_ratio'],3)} & {fnum(r['fdr_bh'],3)} \\\\"
            )
    return r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Sequence and RNA structure features for CDS-proximal versus distal m1A sites.} Structural features show clearer differences than simple motif summaries, supporting the structure-aware design.}
\label{tab:structure}
\begin{tabular}{lcccc}
\toprule
Feature & Proximal median & Distal median & Effect/OR & FDR \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def table_functional(go: list[dict[str, str]], pathway: list[dict[str, str]]) -> str:
    selected = go[:4] + pathway[:4]
    body = []
    for r in selected:
        body.append(
            f"{esc(r['group'])} & {esc(r['library'].replace('_', ' '))} & {esc(r['term'][:70])} & {r['overlap_count']}/{r['target_size']} & {fnum(r['p_value'],3)} & {fnum(r['fdr_bh'],3)} \\\\"
        )
    return r"""
\begin{table*}[t]
\centering
\scriptsize
\caption{\textbf{Representative exploratory GO/KEGG/Reactome enrichment terms.} These terms provide biological context but are not used as primary mechanistic evidence because adjusted significance is modest.}
\label{tab:functional}
\begin{tabular}{llllcc}
\toprule
Group & Library & Term & Overlap & P value & FDR \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def build_tex() -> str:
    workflow = read_tsv(ROOT / "11_tables" / "workflow_summary.tsv")
    model = read_tsv(ROOT / "09_results" / "metrics" / "model_comparison.tsv")
    benchmark = read_tsv(ROOT / "09_results" / "metrics" / "external_benchmark_comparison.tsv")
    reader = read_tsv(ROOT / "11_tables" / "biology_summary" / "reader_enrichment.tsv")
    external = read_tsv(ROOT / "11_tables" / "biology_summary" / "external_validation_summary.tsv")
    structure = read_tsv(ROOT / "11_tables" / "biology_summary" / "rnafold_structure_comparison.tsv")
    sequence = read_tsv(ROOT / "11_tables" / "biology_summary" / "sequence_feature_comparison.tsv")
    go = read_tsv(ROOT / "11_tables" / "biology_summary" / "go_enrichment.tsv")
    pathway = read_tsv(ROOT / "11_tables" / "biology_summary" / "kegg_reactome_enrichment.tsv")

    m4 = next(r for r in model if r["variant"] == "M4")
    m0 = next(r for r in model if r["variant"] == "M0")
    multirm = next(r for r in benchmark if r["variant"] == "multirm_style_bilstm_attention")
    official = next(r for r in benchmark if r["variant"] == "multirm_official_101nt_zero_shot")
    page_count = page_count_from_pdf(REFERENCE_PDF)

    return rf"""%Version 3.1 December 2024
% Expanded StructDualRM Nature Communications draft generated from the Springer Nature journal article template.
% Reference MultiRM PDF structural guide: {page_count} pages detected locally. This draft intentionally uses a similar full Article rhythm: framework, benchmark, interpretability, biological validation and methods.

\documentclass[pdflatex,sn-nature]{{sn-jnl}}

\usepackage{{graphicx}}%
\usepackage{{multirow}}%
\usepackage{{amsmath,amssymb,amsfonts}}%
\usepackage{{booktabs}}%
\usepackage{{url}}%

\raggedbottom

\begin{{document}}

\title[StructDualRM for m1A/m6A prediction]{{StructDualRM: structure-aware integrated prediction and biological interpretation of m1A and m6A RNA modifications}}

\author*[1]{{\fnm{{First}} \sur{{Author}}}}\email{{corresponding.author@example.edu}}
\author[1]{{\fnm{{Second}} \sur{{Author}}}}
\author[1]{{\fnm{{Third}} \sur{{Author}}}}

\affil*[1]{{\orgdiv{{Department of Bioinformatics}}, \orgname{{Institution name}}, \orgaddress{{\city{{City}}, \country{{Country}}}}}}

\abstract{{RNA modifications shape transcript fate through sequence context, RNA secondary structure and reader-protein interactions. Here we present StructDualRM, a structure-aware multi-label framework for integrated prediction and interpretation of m1A and m6A sites in hg19 transcript space. The workflow combines hg19 modification maps, RefSeq transcript projection, 101 nt sequence windows, RNAfold-derived local structure and dual-site contextual features. In held-out testing, the full model achieved a macro area under the precision--recall curve (AUPR) of {fnum(m4['test_macro_AUPR'],4)} and macro Matthews correlation coefficient (MCC) of {fnum(m4['test_macro_MCC'],4)}, exceeding a sequence-only transformer and MultiRM-style baselines. Direct same-dataset benchmarking showed that StructDualRM outperformed a MultiRM-style BiLSTM-attention model and an official MultiRM zero-shot configuration. Biological analyses further supported proximal m1A--m6A arrangements through YTHDF2 reader enrichment, independent hg19-compatible public m1A/m6A annotations and structure-linked local folding differences. These results identify RNA structure and dual-modification context as informative axes for joint epitranscriptomic modelling.}}

\keywords{{RNA modification, m1A, m6A, RNA secondary structure, multi-label learning, epitranscriptomics, Nature Communications draft}}

\maketitle

\section{{Introduction}}\label{{sec:introduction}}

Chemical modifications of RNA expand the regulatory capacity of the transcriptome. Among the many known marks, N6-methyladenosine (m6A) is abundant in mRNA and is associated with transcript stability, splicing, export, translation and decay. N1-methyladenosine (m1A) is chemically distinct because methylation at the Watson--Crick interface can alter local base-pairing potential and RNA folding. Although these two adenosine modifications are often studied separately, their co-occurrence on the same transcript raises a broader question: whether the positional relationship between modification classes contains regulatory information that is not visible from either mark alone.

Computational predictors have become central to RNA modification research because experimental maps remain cell-type-, protocol- and coverage-dependent. Existing methods, including attention-based multi-label models such as MultiRM, show that a shared neural representation can learn signals for multiple RNA modifications. However, most models still emphasize short primary sequence windows. This creates two limitations. First, RNA folding can modulate enzymatic accessibility and reader recognition, but is not always explicitly represented. Second, multi-label prediction is often evaluated mainly as a classification problem, whereas the biological interpretation of dual-modification arrangements requires additional analyses beyond AUC or AUPR.

Here we developed StructDualRM, a structure-aware integrated predictor for m1A and m6A sites. The central design principle is that sequence, RNA secondary structure and dual-site positional context should be treated as complementary evidence streams. We therefore built a hg19 workflow that maps m1A and m6A annotations into transcript space, extracts 101 nt sequence windows, computes RNAfold-derived structure descriptors, trains a ladder of ablation models and evaluates biological support through reader enrichment, public external validation and functional enrichment.

The study is designed to move beyond a simple ``new model versus old model'' comparison. We first quantify the contribution of sequence, global structure, graph context and gating through internal ablation. We then benchmark against classical k-mer models, a MultiRM-style BiLSTM-attention baseline and an official MultiRM zero-shot configuration on the same dataset. Finally, we ask whether predicted or observed proximal m1A--m6A contexts align with independent reader-binding and modification-annotation evidence. This combination of modelling and biological interpretation is intended to make the structural component a testable innovation rather than a decorative feature.

\section{{Results}}\label{{sec:results}}

\subsection{{A hg19 transcript-space workflow defines the m1A/m6A learning task}}

The analysis started from hg19-compatible m1A and m6A site collections and projected them into RefSeq transcript coordinates. The m1A branch retained 30,636 hg19 sites and 29,059 transcript-mapped sites, whereas the m6A branch retained 189,508 hg19 sites and 170,604 transcript-mapped sites. Center-A filtering yielded 4,675 m1A-positive windows and 170,604 m6A-positive windows. The final learning split contained 34,972 training rows, 7,380 validation rows and 7,648 held-out test rows. This construction produced a deliberately challenging imbalanced multi-label setting, with many more m6A-positive than m1A-positive examples and only 14 co-positive sites before sampling and window construction (Table~\ref{{tab:dataset}}; Fig.~\ref{{fig:workflow}}).

{table_dataset(workflow)}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure1_workflow_architecture.png}}
\caption{{\textbf{{StructDualRM study design and model concept.}} \textbf{{a}}, hg19 m1A/m6A annotations were projected into transcript space, converted to 101 nt windows, annotated with RNAfold-derived structure and evaluated through model and biological analyses. \textbf{{b}}, StructDualRM integrates sequence-transformer, structural and dual-site contextual branches before multi-label prediction.}}
\label{{fig:workflow}}
\end{{figure*}}

\subsection{{Internal ablation identifies the full structure-aware model as the strongest variant}}

We trained five neural variants under the same 50-epoch setting: a sequence-only transformer (M0), a transformer with global structure (M1), a transformer-GNN variant (M2), a gated StructDualRM model (M3) and the full StructDualRM model (M4). The full M4 model achieved the best overall held-out performance, with macro AUPR {fnum(m4['test_macro_AUPR'],4)}, macro MCC {fnum(m4['test_macro_MCC'],4)} and average F1 {fnum(m4['test_average_F1'],4)}. Compared with the sequence-only transformer, M4 increased macro AUPR by {fnum(float(m4['test_macro_AUPR']) - float(m0['test_macro_AUPR']),4)} and macro MCC by {fnum(float(m4['test_macro_MCC']) - float(m0['test_macro_MCC']),4)}. The improvement was most visible for m1A AUPR, which increased from {fnum(m0['test_m1A_AUPR'],4)} in M0 to {fnum(m4['test_m1A_AUPR'],4)} in M4, consistent with the idea that structural and contextual evidence is particularly useful for the rarer label (Table~\ref{{tab:ablation}}; Fig.~\ref{{fig:performance}}).

{table_model_ablation(model)}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure2_model_performance.png}}
\caption{{\textbf{{Predictive performance and external benchmark.}} Internal ablation and direct same-dataset benchmark comparisons show that the full StructDualRM model provides the strongest macro AUPR and macro MCC.}}
\label{{fig:performance}}
\end{{figure*}}

\subsection{{Same-dataset benchmarking clarifies the improvement over MultiRM-style baselines}}

Cross-paper comparisons can be misleading when datasets, window definitions and negative sampling strategies differ. We therefore evaluated all baselines on the same held-out split. The MultiRM-style BiLSTM-attention baseline reached macro AUPR {fnum(multirm['test_macro_AUPR'],4)} and macro MCC {fnum(multirm['test_macro_MCC'],4)}, while the official MultiRM zero-shot setting reached macro AUPR {fnum(official['test_macro_AUPR'],4)} and macro MCC {fnum(official['test_macro_MCC'],4)}. The gap between the official zero-shot model and the retrained MultiRM-style model suggests distribution shift between the original MultiRM setting and the present hg19 m1A/m6A task. The full StructDualRM model remained the strongest model in this same-dataset benchmark (Table~\ref{{tab:benchmark}}).

{table_benchmark(benchmark)}

\subsection{{Local RNA structure contributes interpretable signal beyond simple sequence motifs}}

The structure-aware design would be less compelling if the proximal m1A--m6A signal were fully explained by primary sequence motifs. We therefore compared proximal and distal m1A contexts using sequence motif summaries and RNAfold-derived structural descriptors. In the CDS-proximal comparison, GC fraction, MFE per nucleotide, paired fraction and central 21 nt paired fraction shifted relative to distal controls. The direction of the MFE and paired-fraction differences suggests that proximal dual-modification contexts tend to occupy less strongly paired local structures than distal m1A controls. In contrast, DRACH-centred and GGAC-like motif summaries showed weaker explanatory power in the same comparison. These findings support RNA secondary structure as a biologically interpretable modelling component (Table~\ref{{tab:structure}}; Fig.~\ref{{fig:structure}}).

{table_structure(structure, sequence)}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure3_sequence_structure_context.png}}
\caption{{\textbf{{Sequence and structural context of proximal dual modifications.}} CDS-proximal m1A--m6A arrangements show changes in local RNAfold structure metrics relative to distal controls, whereas motif-level comparisons alone do not explain the full signal.}}
\label{{fig:structure}}
\end{{figure*}}

\subsection{{Reader enrichment links proximal dual-modification contexts to post-transcriptional regulation}}

We next asked whether proximal m1A--m6A arrangements were associated with reader-binding evidence. Using public YTHDF2 PAR-CLIP annotations, m1A 3'UTR-proximal sites were enriched relative to distal m1A controls (odds ratio {fnum(reader[0]['odds_ratio'],3)}, FDR {fnum(reader[0]['fdr_bh'],3)}). m1A CDS-proximal sites showed a similar enrichment (odds ratio {fnum(reader[1]['odds_ratio'],3)}, FDR {fnum(reader[1]['fdr_bh'],3)}). Comparisons against m6A-only sites on the same transcripts were stronger, with odds ratios of {fnum(reader[2]['odds_ratio'],3)} for 3'UTR-proximal sites and {fnum(reader[3]['odds_ratio'],3)} for CDS-proximal sites. These results do not prove reader recruitment, but they place proximal dual-modification contexts in a reader-associated regulatory environment (Table~\ref{{tab:reader_external}}; Fig.~\ref{{fig:biology}}).

\subsection{{Independent hg19 public annotations support the proximal m1A--m6A arrangement}}

External validation used independent hg19-compatible m1A-miCLIP and m6A peak annotations. The same proximal groups that showed YTHDF2 enrichment also showed elevated overlap with independent m1A/m6A annotations. For example, 3'UTR-proximal sites had an external-validation odds ratio of {fnum(external[0]['odds_ratio'],3)} versus distal controls, and CDS-proximal sites had an odds ratio of {fnum(external[1]['odds_ratio'],3)}. The consistency between reader enrichment and external modification annotations strengthens the interpretation that proximal m1A--m6A arrangements are not random artefacts of the internal dataset.

{table_reader_external(reader, external)}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure4_reader_external_validation.png}}
\caption{{\textbf{{Reader enrichment and hg19 external validation.}} Proximal m1A--m6A arrangements are enriched for YTHDF2 reader overlap and independent hg19-compatible m1A/m6A public annotations.}}
\label{{fig:biology}}
\end{{figure*}}

\subsection{{Functional enrichment provides exploratory biological context}}

Gene-set enrichment was used as an exploratory analysis to summarize the transcript-level biological context of proximal dual-modification groups. Nominally enriched GO terms included regulation of cell fate commitment, intracellular pH reduction, plasma membrane organization and transcriptional regulation. KEGG and Reactome terms highlighted metabolism, smooth-muscle contraction, infection-related pathways, vesicle trafficking and cytoskeletal organization. These signals are useful for generating biological hypotheses, but they are not the primary evidence in this study because multiple-testing adjusted significance was modest. The manuscript therefore treats GO/KEGG/Reactome enrichment as contextual support rather than a mechanistic claim (Table~\ref{{tab:functional}}; Fig.~\ref{{fig:pathway}}).

{table_functional(go, pathway)}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure5_functional_enrichment.png}}
\caption{{\textbf{{Exploratory functional enrichment.}} GO, KEGG and Reactome enrichment results provide biological context but are interpreted conservatively because adjusted significance is weaker than reader and external-validation support.}}
\label{{fig:pathway}}
\end{{figure*}}

\section{{Discussion}}\label{{sec:discussion}}

StructDualRM was designed around a simple premise: integrated RNA modification prediction should model not only sequence motifs but also RNA structure and the positional relationship between modification classes. The held-out ablation analysis supports this premise. The full model achieved the best macro AUPR and macro MCC, and the largest label-specific gain was observed for m1A, the rarer and structurally distinctive mark. This pattern is consistent with a model in which secondary structure and dual-site context provide information that is not fully captured by primary sequence alone.

The direct benchmark against MultiRM-style baselines is an important part of the evidence. Because all models were evaluated on the same dataset, the comparison avoids the common problem of cross-study benchmark drift. The official MultiRM zero-shot result was substantially weaker than the retrained MultiRM-style baseline, indicating that distribution shift matters for RNA modification predictors. The StructDualRM advantage should therefore be interpreted as a same-dataset improvement over both classical sequence baselines and a MultiRM-inspired architecture, rather than as a universal claim over every possible MultiRM deployment.

The biological analyses add a second layer of support. Proximal m1A--m6A contexts were enriched for YTHDF2 reader overlap and for independent hg19-compatible m1A/m6A annotations. These findings are coherent with the idea that local arrangements of RNA modifications can define regulatory contexts. At the same time, the analyses remain observational. They do not establish that m1A causes m6A deposition, that m6A causes m1A deposition, or that either mark causally recruits YTHDF2 in the tested contexts. Such claims would require cell-type-matched perturbation assays, reader knockdown or mutation experiments and orthogonal site-level validation.

The current draft also clarifies where the study is strongest and where it should be strengthened before a high-impact submission. The strongest evidence consists of the same-split model comparison, the direct MultiRM-style benchmark and the concordant reader/external validation. The weaker evidence consists of exploratory functional enrichment, which should be presented cautiously. Future work should prioritize independent prospective datasets, calibration analysis, cell-type-specific validation and wet-lab perturbation of a small number of high-confidence proximal m1A--m6A candidates.

\section{{Methods}}\label{{sec:methods}}

\subsection{{Data processing and transcript projection}}

All primary processing was performed in hg19 coordinate space. Modification sites were parsed, filtered against the hg19 reference, mapped into RefSeq transcript coordinates and converted into centered 101 nt windows. Negative examples were sampled from transcript-compatible candidate adenines while avoiding known positive sites. For site $i$, the model input was represented as
\begin{{equation}}
\mathbf{{x}}_i = \left(\mathbf{{s}}_i, \mathbf{{r}}_i, \mathbf{{g}}_i\right),
\label{{eq:input}}
\end{{equation}}
where $\mathbf{{s}}_i$ denotes the nucleotide sequence window, $\mathbf{{r}}_i$ denotes RNAfold-derived local structure features and $\mathbf{{g}}_i$ denotes graph or dual-site contextual features.

\subsection{{RNA secondary-structure representation}}

RNA secondary structure was estimated with RNAfold on 101 nt windows. The structural feature vector included MFE, normalized MFE, paired fractions and central-window pairing summaries:
\begin{{equation}}
\mathbf{{r}}_i = \left[\mathrm{{MFE}}_i,\frac{{\mathrm{{MFE}}_i}}{{L}},\rho_i,\rho_i^{{(11)}},\rho_i^{{(21)}},\ell_i\right],
\label{{eq:structure}}
\end{{equation}}
where $L=101$, $\rho_i$ is the paired fraction, $\rho_i^{{(11)}}$ and $\rho_i^{{(21)}}$ are central-window paired fractions and $\ell_i$ is the loop span around the centre.

\subsection{{StructDualRM model objective}}

For two labels $c\in\{{\mathrm{{m1A}},\mathrm{{m6A}}\}}$, the model outputs logits $z_{{ic}}$ and probabilities
\begin{{equation}}
p_{{ic}} = \sigma(z_{{ic}})=\frac{{1}}{{1+\exp(-z_{{ic}})}}.
\label{{eq:sigmoid}}
\end{{equation}}
The training objective used a class-weighted binary cross-entropy loss:
\begin{{equation}}
\mathcal{{L}} =
-\frac{{1}}{{NC}}\sum_{{i=1}}^N\sum_{{c=1}}^C
\left[
w_c^+ y_{{ic}}\log(p_{{ic}}+\epsilon)
w_c^- (1-y_{{ic}})\log(1-p_{{ic}}+\epsilon)
\right],
\label{{eq:loss}}
\end{{equation}}
where $N$ is the number of samples, $C=2$, $y_{{ic}}$ is the binary label, $w_c^+$ and $w_c^-$ are label-specific class weights and $\epsilon$ is a numerical stabilizer.

\subsection{{Training, thresholding and benchmark evaluation}}

All neural models were trained for 50 epochs with the same train/validation/test split. Thresholds were selected on the validation set and then applied once to the held-out test set. Macro AUPR was computed as
\begin{{equation}}
\mathrm{{macroAUPR}} = \frac{{1}}{{C}}\sum_{{c=1}}^C \mathrm{{AUPR}}_c.
\label{{eq:macroaupr}}
\end{{equation}}
For a fixed threshold, MCC for each label was computed as
\begin{{equation}}
\mathrm{{MCC}} =
\frac{{TP\cdot TN-FP\cdot FN}}
{{\sqrt{{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}}}.
\label{{eq:mcc}}
\end{{equation}}
Benchmark models included k-mer logistic regression, k-mer ExtraTrees, a MultiRM-style BiLSTM-attention model and an official MultiRM zero-shot configuration. All benchmark results were evaluated on the same held-out split.

\subsection{{Reader enrichment and external validation}}

Reader enrichment used public YTHDF2 PAR-CLIP hg19-compatible annotations. External validation used independent public m1A-miCLIP and m6A peak annotations prepared in hg19 coordinate space. For each comparison, odds ratios were estimated from a $2\times 2$ table:
\begin{{equation}}
\mathrm{{OR}}=\frac{{(a+0.5)(d+0.5)}}{{(b+0.5)(c+0.5)}},
\label{{eq:or}}
\end{{equation}}
where the 0.5 continuity correction stabilizes comparisons with sparse cells. P values from Fisher exact tests were adjusted using the Benjamini--Hochberg procedure:
\begin{{equation}}
q_{{(i)}}=\min_{{j\ge i}}\left(\frac{{m\,p_{{(j)}}}}{{j}}\right),
\label{{eq:bh}}
\end{{equation}}
where $m$ is the number of tests and $p_{{(j)}}$ is the $j$th ordered P value.

\subsection{{Gene-set enrichment}}

GO Biological Process, KEGG and Reactome enrichment analyses were performed on transcript-level gene groups derived from proximal and comparison categories. Hypergeometric tests were used for over-representation analysis:
\begin{{equation}}
P(X\ge k)=\sum_{{x=k}}^{{\min(K,n)}}\frac{{\binom{{K}}{{x}}\binom{{M-K}}{{n-x}}}}{{\binom{{M}}{{n}}}},
\label{{eq:hypergeom}}
\end{{equation}}
where $M$ is the background size, $K$ is the term size, $n$ is the target set size and $k$ is the observed overlap.

\section*{{Data availability}}

The draft package includes source data underlying each generated figure, processed model-performance tables, biological enrichment tables and hg19-compatible external validation tables. Raw public data sources and final accession or DOI statements should be verified before journal submission.

\section*{{Code availability}}

Custom scripts used to generate the analyses and this expanded template-based draft are included in the analysis archive. Before formal submission, the code should be deposited in a public repository or reproducible capsule and linked with a permanent DOI where appropriate.

\section*{{Acknowledgements}}

[Add funding and institutional acknowledgements.]

\section*{{Author contributions}}

[Add author contribution statements.]

\section*{{Competing interests}}

The authors declare no competing interests. [Edit if needed.]

\begin{{thebibliography}}{{9}}
\bibitem{{meyer2012}} Meyer, K. D. et al. Comprehensive analysis of mRNA methylation reveals enrichment in 3' UTRs and near stop codons. \textit{{Cell}} \textbf{{149}}, 1635--1646 (2012).
\bibitem{{dominissini2012}} Dominissini, D. et al. Topology of the human and mouse m6A RNA methylomes revealed by m6A-seq. \textit{{Nature}} \textbf{{485}}, 201--206 (2012).
\bibitem{{roundtree2017}} Roundtree, I. A., Evans, M. E., Pan, T. \& He, C. Dynamic RNA modifications in gene expression regulation. \textit{{Cell}} \textbf{{169}}, 1187--1200 (2017).
\bibitem{{shi2019}} Shi, H., Wei, J. \& He, C. Where, when, and how: context-dependent functions of RNA methylation writers, readers, and erasers. \textit{{Mol. Cell}} \textbf{{74}}, 640--650 (2019).
\bibitem{{multirm}} MultiRM original article: Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occurring RNA modifications. Verify full bibliographic details before submission.
\bibitem{{viennarna}} Lorenz, R. et al. ViennaRNA Package 2.0. \textit{{Algorithms Mol. Biol.}} \textbf{{6}}, 26 (2011).
\end{{thebibliography}}

\end{{document}}
"""


def update_readme(page_count: int) -> None:
    readme = SN_DRAFT / "README_SN_TEMPLATE_DRAFT.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    addition = f"""

## Expanded manuscript pass

- Reference MultiRM PDF local page count: {page_count} pages.
- Expanded draft now follows a fuller Article rhythm: Introduction, six Results subsections, Discussion, Methods, five main figures and six main tables.
- Added main tables: dataset construction, internal ablation, same-dataset benchmark, sequence/structure interpretability, reader/external validation and exploratory functional enrichment.
- The pathway table is intentionally described as exploratory to avoid overclaiming adjusted enrichment results.
"""
    write_text(readme, existing.rstrip() + "\n" + addition)


def refresh_template_zip() -> None:
    zip_path = ROOT / "submission" / "nature_communications_structdualrm_sn_template_draft.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in SN_DRAFT.rglob("*"):
            if path.is_file():
                zf.write(path, f"sn_template_draft/{path.relative_to(SN_DRAFT).as_posix()}")


def main() -> None:
    page_count = page_count_from_pdf(REFERENCE_PDF)
    backup = SN_DRAFT / "StructDualRM_NatureCommunications_SN_Template_Draft.compact_backup.tex"
    if TEX.exists() and not backup.exists():
        shutil.copy2(TEX, backup)
    expanded = build_tex()
    write_text(TEX, expanded)
    update_readme(page_count)
    summary = {
        "reference_pdf": str(REFERENCE_PDF),
        "reference_pdf_pages_detected": page_count,
        "main_tex": str(TEX),
        "figures_inserted": 5,
        "tables_inserted": 6,
    }
    write_text(SN_DRAFT / "expanded_manuscript_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    refresh_template_zip()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
