from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "submission" / "nature_communications_structdualrm"
USER_TEMPLATE = Path(r"C:\Users\夏添\Downloads\Download+the+journal+article+template+package+(December+2024+version).zip")
TEMPLATE_EXTRACT = PKG / "template_from_user"
SN_DRAFT = PKG / "sn_template_draft"
NS = "{http://www.w3.org/2000/svg}"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def fmt(x: str | float, digits: int = 4) -> str:
    value = float(x)
    if abs(value) >= 1:
        return f"{value:.3f}" if digits <= 3 else f"{value:.4f}"
    if abs(value) >= 0.001:
        return f"{value:.{digits}f}"
    return f"{value:.2e}"


def tex_escape(text: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf") if bold else Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def color(value: str | None, default: str = "#1f2933") -> str:
    if value in (None, "", "none"):
        return default
    return value


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value.replace("px", ""))


def parse_length(value: str | None, canvas: int, default: float = 0.0) -> float:
    if value is None:
        return default
    if value.endswith("%"):
        return canvas * float(value[:-1]) / 100.0
    return parse_float(value, default)


def draw_arrow(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, fill: str) -> None:
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 16
    spread = 0.42
    p1 = (x2, y2)
    p2 = (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread))
    p3 = (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread))
    draw.polygon([p1, p2, p3], fill=fill)


def text_style(cls: str | None) -> tuple[int, bool]:
    if cls == "title":
        return 26, True
    if cls == "label":
        return 20, True
    if cls == "tiny":
        return 14, False
    return 18, False


def render_svg_to_png(svg_path: Path, png_path: Path) -> None:
    root = ET.parse(svg_path).getroot()
    width = int(parse_float(root.attrib.get("width"), 1800))
    height = int(parse_float(root.attrib.get("height"), 1120))
    scale = 2
    img = Image.new("RGB", (width * scale, height * scale), "#fbfcfe")
    draw = ImageDraw.Draw(img)

    def sx(v: float) -> float:
        return v * scale

    def sw(v: float) -> int:
        return max(1, int(v * scale))

    for elem in root.iter():
        tag = elem.tag.replace(NS, "")
        if tag in {"svg", "defs", "style", "marker", "path"}:
            continue
        if tag == "rect":
            x = sx(parse_float(elem.attrib.get("x")))
            y = sx(parse_float(elem.attrib.get("y")))
            w = sx(parse_length(elem.attrib.get("width"), width))
            h = sx(parse_length(elem.attrib.get("height"), height))
            fill = color(elem.attrib.get("fill"), "#ffffff")
            outline = elem.attrib.get("stroke")
            stroke_width = sw(parse_float(elem.attrib.get("stroke-width"), 1))
            draw.rounded_rectangle([x, y, x + w, y + h], radius=sw(parse_float(elem.attrib.get("rx"), 0)), fill=fill, outline=outline, width=stroke_width if outline else 1)
        elif tag == "line":
            x1, y1 = sx(parse_float(elem.attrib.get("x1"))), sx(parse_float(elem.attrib.get("y1")))
            x2, y2 = sx(parse_float(elem.attrib.get("x2"))), sx(parse_float(elem.attrib.get("y2")))
            stroke = color(elem.attrib.get("stroke"), "#44515e")
            width_line = sw(parse_float(elem.attrib.get("stroke-width"), 2))
            draw.line([x1, y1, x2, y2], fill=stroke, width=width_line)
            if "marker-end" in elem.attrib:
                draw_arrow(draw, x1, y1, x2, y2, stroke)
        elif tag == "circle":
            cx, cy = sx(parse_float(elem.attrib.get("cx"))), sx(parse_float(elem.attrib.get("cy")))
            r = sx(parse_float(elem.attrib.get("r")))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color(elem.attrib.get("fill"), "#1f2933"), outline=elem.attrib.get("stroke"))
        elif tag == "text":
            txt = "".join(elem.itertext())
            x, y = sx(parse_float(elem.attrib.get("x"))), sx(parse_float(elem.attrib.get("y")))
            size, bold = text_style(elem.attrib.get("class"))
            fnt = font(size * scale, bold)
            fill = color(elem.attrib.get("fill"), "#1f2933")
            bbox = draw.textbbox((0, 0), txt, font=fnt)
            tw = bbox[2] - bbox[0]
            anchor = elem.attrib.get("text-anchor")
            if anchor in {"middle", "center"}:
                x -= tw / 2
            elif anchor == "end":
                x -= tw
            draw.text((x, y - size * scale), txt, font=fnt, fill=fill)

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)


def prepare_template() -> None:
    if not (TEMPLATE_EXTRACT / "sn-article-template" / "sn-jnl.cls").exists():
        if TEMPLATE_EXTRACT.exists():
            shutil.rmtree(TEMPLATE_EXTRACT)
        TEMPLATE_EXTRACT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(USER_TEMPLATE) as zf:
            zf.extractall(TEMPLATE_EXTRACT)
    SN_DRAFT.mkdir(parents=True, exist_ok=True)
    src = TEMPLATE_EXTRACT / "sn-article-template"
    for name in ["sn-jnl.cls", "user-manual.pdf", "sn-article.pdf"]:
        shutil.copy2(src / name, SN_DRAFT / name)
    bst_dst = SN_DRAFT / "bst"
    if bst_dst.exists():
        shutil.rmtree(bst_dst)
    shutil.copytree(src / "bst", bst_dst)
    shutil.copy2(src / "bst" / "sn-nature.bst", SN_DRAFT / "sn-nature.bst")


def make_png_figures() -> list[str]:
    names = [
        "Figure1_workflow_architecture",
        "Figure2_model_performance",
        "Figure3_sequence_structure_context",
        "Figure4_reader_external_validation",
        "Figure5_functional_enrichment",
    ]
    for name in names:
        render_svg_to_png(PKG / "figures" / f"{name}.svg", PKG / "figures_png" / f"{name}.png")
        shutil.copy2(PKG / "figures_png" / f"{name}.png", SN_DRAFT / f"{name}.png")
    return names


def make_template_tex() -> None:
    model = read_tsv(ROOT / "09_results" / "metrics" / "model_comparison.tsv")
    bench = read_tsv(ROOT / "09_results" / "metrics" / "external_benchmark_comparison.tsv")
    reader = read_tsv(ROOT / "11_tables" / "biology_summary" / "reader_enrichment.tsv")
    external = read_tsv(ROOT / "11_tables" / "biology_summary" / "external_validation_summary.tsv")
    m4 = next(r for r in model if r["variant"] == "M4")
    m0 = next(r for r in model if r["variant"] == "M0")
    multirm = next(r for r in bench if r["variant"] == "multirm_style_bilstm_attention")
    official = next(r for r in bench if r["variant"] == "multirm_official_101nt_zero_shot")

    tex = rf"""%Version 3.1 December 2024
% StructDualRM Nature Communications draft generated from the Springer Nature journal article template package.

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

\abstract{{RNA modifications shape transcript fate through sequence context, RNA secondary structure and reader-protein interactions. Here we present StructDualRM, a structure-aware multi-label framework for integrated prediction of m1A and m6A sites in hg19 transcript space. The workflow combines hg19 modification maps, RefSeq transcript projection, 101 nt sequence windows, RNAfold-derived local structure and dual-site contextual features. In held-out testing, the full model achieved a macro area under the precision--recall curve (AUPR) of {fmt(m4['test_macro_AUPR'])} and macro Matthews correlation coefficient (MCC) of {fmt(m4['test_macro_MCC'])}, exceeding a sequence-only transformer and MultiRM-style baselines. Direct same-dataset benchmarking showed that StructDualRM outperformed a MultiRM-style BiLSTM-attention model and an official MultiRM zero-shot configuration. Biological analyses further supported proximal m1A--m6A arrangements through YTHDF2 reader enrichment and independent hg19-compatible public m1A/m6A annotations. These results identify RNA structure and dual-modification context as informative axes for joint epitranscriptomic modelling.}}

\keywords{{RNA modification, m1A, m6A, RNA secondary structure, multi-label learning, epitranscriptomics, Nature Communications draft}}

\maketitle

\section{{Introduction}}\label{{sec:introduction}}

Post-transcriptional RNA modifications regulate transcript stability, localization, translation and cellular adaptation. N6-methyladenosine (m6A) is the most extensively studied internal mRNA modification, whereas N1-methyladenosine (m1A) has distinct chemical consequences and positional preferences. Their relationship is not only a question of site-level prediction: the spatial arrangement of modifications on the same transcript may influence reader binding, local RNA folding and downstream regulatory programs.

Existing multi-label models, including MultiRM-like attention models, demonstrate that shared neural representations can predict multiple RNA modification types. However, models that rely mainly on primary sequence may miss structural and positional context. We therefore asked whether m1A and m6A can be predicted jointly while making RNA secondary structure and proximal dual-site context explicit components of the model.

\section{{Results}}\label{{sec:results}}

\subsection{{A hg19 workflow for joint m1A/m6A modelling}}

The dataset was constructed from hg19 modification annotations and projected into RefSeq transcript coordinates. After filtering, the workflow retained 29,059 mapped m1A sites and 170,604 mapped m6A sites. Center-A filtering yielded 4,675 m1A-positive windows and 170,604 m6A-positive windows, with 14 co-positive sites and 186,237 negative candidates. The final supervised learning split contained 34,972 training rows, 7,380 validation rows and 7,648 held-out test rows (Fig.~\ref{{fig:workflow}}).

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure1_workflow_architecture.png}}
\caption{{\textbf{{StructDualRM study design and model concept.}} \textbf{{a}}, hg19 m1A/m6A annotations were projected into transcript space, converted to 101 nt windows, annotated with RNAfold-derived structure and evaluated through model and biological analyses. \textbf{{b}}, StructDualRM integrates sequence-transformer, structural and dual-site contextual branches before multi-label prediction.}}
\label{{fig:workflow}}
\end{{figure*}}

\subsection{{StructDualRM improves held-out multi-label prediction}}

We trained five neural variants under the same split and 50-epoch setting: a sequence-only transformer (M0), a transformer with global structure (M1), a transformer-GNN variant (M2), a gated StructDualRM model (M3) and the full StructDualRM model (M4). M4 achieved the best held-out performance with macro AUPR {fmt(m4['test_macro_AUPR'])}, macro MCC {fmt(m4['test_macro_MCC'])} and average F1 {fmt(m4['test_average_F1'])}. Compared with M0, M4 improved macro AUPR by {fmt(float(m4['test_macro_AUPR']) - float(m0['test_macro_AUPR']))} and macro MCC by {fmt(float(m4['test_macro_MCC']) - float(m0['test_macro_MCC']))} (Fig.~\ref{{fig:performance}}).

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure2_model_performance.png}}
\caption{{\textbf{{Predictive performance and external benchmark.}} Internal ablation and direct same-dataset benchmark comparisons show that the full StructDualRM model provides the strongest macro AUPR and macro MCC.}}
\label{{fig:performance}}
\end{{figure*}}

\subsection{{Direct comparison with MultiRM-style baselines}}

To test whether the improvement reflected architecture rather than favourable comparison groups, we evaluated classical k-mer models, a MultiRM-style BiLSTM-attention model and an official MultiRM zero-shot configuration on the same held-out dataset. StructDualRM M4 outperformed the MultiRM-style BiLSTM-attention model (macro AUPR {fmt(multirm['test_macro_AUPR'])}; macro MCC {fmt(multirm['test_macro_MCC'])}) and the official MultiRM zero-shot setting (macro AUPR {fmt(official['test_macro_AUPR'])}; macro MCC {fmt(official['test_macro_MCC'])}).

\subsection{{Structure-aware modelling is supported by local sequence-structure analyses}}

The proximal m1A--m6A signal was not reducible to canonical motif presence alone. In CDS-proximal comparisons, local folding metrics shifted relative to distal m1A controls, including MFE per nucleotide, paired fraction and central paired fraction. These analyses support the structural branch as a mechanistic innovation point while remaining correlative and computational (Fig.~\ref{{fig:structure}}).

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure3_sequence_structure_context.png}}
\caption{{\textbf{{Sequence and structural context of proximal dual modifications.}} CDS-proximal m1A--m6A arrangements show changes in local RNAfold structure metrics relative to distal controls, whereas motif-level comparisons alone do not explain the full signal.}}
\label{{fig:structure}}
\end{{figure*}}

\subsection{{Proximal m1A--m6A sites are enriched for reader and external annotation support}}

YTHDF2 PAR-CLIP overlap was enriched in proximal m1A--m6A groups. m1A 3'UTR-proximal sites showed an odds ratio of {fmt(reader[0]['odds_ratio'], 3)} versus distal m1A controls (FDR {fmt(reader[0]['fdr_bh'])}), and m1A CDS-proximal sites showed an odds ratio of {fmt(reader[1]['odds_ratio'], 3)} (FDR {fmt(reader[1]['fdr_bh'])}). Independent hg19-compatible m1A/m6A public annotations showed concordant enrichment, with odds ratios of {fmt(external[0]['odds_ratio'], 3)} for 3'UTR-proximal sites and {fmt(external[1]['odds_ratio'], 3)} for CDS-proximal sites against distal controls (Fig.~\ref{{fig:biology}}).

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure4_reader_external_validation.png}}
\caption{{\textbf{{Reader enrichment and hg19 external validation.}} Proximal m1A--m6A arrangements are enriched for YTHDF2 reader overlap and independent hg19-compatible m1A/m6A public annotations.}}
\label{{fig:biology}}
\end{{figure*}}

\subsection{{Exploratory GO/KEGG/Reactome analyses}}

Gene-set analyses provided exploratory biological context for transcripts carrying proximal dual-modification arrangements. Several nominally enriched terms involved chromatin regulation, intracellular acidification, actin/cytoskeleton organization and vesicle-related processes. Because many multiple-testing adjusted values were modest, these pathway results are treated as hypothesis-generating rather than primary mechanistic evidence (Fig.~\ref{{fig:pathway}}).

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{Figure5_functional_enrichment.png}}
\caption{{\textbf{{Exploratory functional enrichment.}} GO, KEGG and Reactome enrichment results provide biological context but are interpreted conservatively because adjusted significance is weaker than reader and external-validation support.}}
\label{{fig:pathway}}
\end{{figure*}}

\section{{Discussion}}\label{{sec:discussion}}

StructDualRM advances integrated RNA modification prediction by bringing RNA secondary structure and proximal dual-modification context into a multi-label prediction framework. The strongest computational evidence is the held-out and direct benchmark performance. The strongest biological support is the concordant reader enrichment and external hg19 annotation overlap. Together, these analyses suggest that the spatial relationship between m1A and m6A sites can encode regulatory information that is missed by purely sequence-level models.

Several limitations remain. First, the biological analyses are computational and correlative; they do not establish causal reader recruitment or experimentally validate individual sites. Second, hg19-compatible public annotations differ in protocol, cell type and detection sensitivity. Third, GO/KEGG/Reactome analyses are best viewed as exploratory. Future work should include cell-type matched perturbation assays, prospective validation on independent sequencing experiments and calibrated uncertainty estimates for site-level predictions.

\section{{Methods}}\label{{sec:methods}}

\subsection{{Data processing}}

All primary processing was performed in hg19 coordinate space. Modification sites were parsed, filtered against the hg19 reference, mapped into RefSeq transcript coordinates and converted into centered 101 nt windows. Negative examples were sampled from transcript-compatible candidate adenines while avoiding known positive sites. For site $i$, the model input was represented as
\begin{{equation}}
\mathbf{{x}}_i = \left(\mathbf{{s}}_i, \mathbf{{r}}_i, \mathbf{{g}}_i\right),
\label{{eq:input}}
\end{{equation}}
where $\mathbf{{s}}_i$ denotes the nucleotide sequence window, $\mathbf{{r}}_i$ denotes RNAfold-derived local structure features, and $\mathbf{{g}}_i$ denotes graph or dual-site contextual features.

RNA secondary structure was estimated with RNAfold on 101 nt windows. The structural feature vector included MFE, normalized MFE, paired fractions and central-window pairing summaries:
\begin{{equation}}
\mathbf{{r}}_i = \left[\mathrm{{MFE}}_i,\frac{{\mathrm{{MFE}}_i}}{{L}},\rho_i,\rho_i^{{(11)}},\rho_i^{{(21)}},\ell_i\right],
\label{{eq:structure}}
\end{{equation}}
where $L=101$, $\rho_i$ is the paired fraction, $\rho_i^{{(11)}}$ and $\rho_i^{{(21)}}$ are central-window paired fractions and $\ell_i$ is the loop span around the centre.

\subsection{{Multi-label model objective}}

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

\subsection{{Model evaluation}}

All neural models were trained for 50 epochs with the same train/validation/test split. Performance was evaluated using label-specific AUC, AUPR, F1, MCC, recall and specificity. Macro AUPR was computed as
\begin{{equation}}
\mathrm{{macroAUPR}} = \frac{{1}}{{C}}\sum_{{c=1}}^C \mathrm{{AUPR}}_c.
\label{{eq:macroaupr}}
\end{{equation}}
For a fixed validation-selected threshold, MCC for each label was computed as
\begin{{equation}}
\mathrm{{MCC}} =
\frac{{TP\cdot TN-FP\cdot FN}}
{{\sqrt{{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}}}.
\label{{eq:mcc}}
\end{{equation}}

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

Custom scripts used to generate the analyses and this template-based draft are included in the analysis archive. Before formal submission, the code should be deposited in a public repository or reproducible capsule and linked with a permanent DOI where appropriate.

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
    write_text(SN_DRAFT / "StructDualRM_NatureCommunications_SN_Template_Draft.tex", tex)

    md_note = """# Springer Nature template-based StructDualRM draft

This folder was generated from the user-provided December 2024 Springer Nature journal article template package.

Main file:

- `StructDualRM_NatureCommunications_SN_Template_Draft.tex`

Inserted figures:

- `Figure1_workflow_architecture.png`
- `Figure2_model_performance.png`
- `Figure3_sequence_structure_context.png`
- `Figure4_reader_external_validation.png`
- `Figure5_functional_enrichment.png`

Added formula locations:

- Input representation: Eq. 1
- RNAfold structural vector: Eq. 2
- Sigmoid output: Eq. 3
- Weighted binary cross-entropy: Eq. 4
- Macro AUPR: Eq. 5
- MCC: Eq. 6
- Odds ratio: Eq. 7
- Benjamini-Hochberg FDR: Eq. 8
- Hypergeometric enrichment: Eq. 9
"""
    write_text(SN_DRAFT / "README_SN_TEMPLATE_DRAFT.md", md_note)


def update_package_zip() -> None:
    zip_path = ROOT / "submission" / "nature_communications_structdualrm_full_package.zip"
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in (PKG / "figures_png").glob("*.png"):
            zf.write(path, f"nature_communications_structdualrm/figures_png/{path.name}")
        for path in SN_DRAFT.rglob("*"):
            if path.is_file():
                zf.write(path, f"nature_communications_structdualrm/sn_template_draft/{path.relative_to(SN_DRAFT).as_posix()}")


def create_template_zip() -> Path:
    zip_path = ROOT / "submission" / "nature_communications_structdualrm_sn_template_draft.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in SN_DRAFT.rglob("*"):
            if path.is_file():
                zf.write(path, f"sn_template_draft/{path.relative_to(SN_DRAFT).as_posix()}")
    return zip_path


def main() -> None:
    prepare_template()
    figures = make_png_figures()
    make_template_tex()
    summary = {
        "sn_template_draft": str(SN_DRAFT),
        "template_zip": str(USER_TEMPLATE),
        "png_figures": figures,
        "main_tex": str(SN_DRAFT / "StructDualRM_NatureCommunications_SN_Template_Draft.tex"),
    }
    write_text(SN_DRAFT / "sn_template_generation_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    template_zip = create_template_zip()
    summary["template_draft_zip"] = str(template_zip)
    write_text(SN_DRAFT / "sn_template_generation_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    update_package_zip()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
