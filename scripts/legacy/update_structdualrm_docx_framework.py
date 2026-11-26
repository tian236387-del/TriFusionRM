#!/usr/bin/env python
"""Patch the StructDualRM Chinese manuscript DOCX with fuller framework details."""

from __future__ import annotations

import copy
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
INPUT_DOCX = Path(r"C:\Users\夏添\Downloads\StructDualRM_正文部分中文翻译.docx")
OUTPUT_DOCX = ROOT / "StructDualRM_正文部分中文翻译_框架细节补充版.docx"
BACKUP_DOCX = ROOT / "StructDualRM_正文部分中文翻译_原始备份.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


INTRO_FRAMEWORK = (
    "在此，我们提出StructDualRM，这是一个结构感知的多标签框架，用于在人类hg19转录本空间中对m1A和m6A修饰位点进行整合预测和解释。"
    "与仅把序列窗口压缩为k-mer组成或单一深度特征的做法不同，StructDualRM从数据层开始即围绕“中心腺苷是否被修饰”定义学习任务："
    "所有候选位点先被统一到hg19兼容坐标，映射到RefSeq转录本坐标，再提取以候选A为中心的101 nt窗口，并同时记录5′UTR、CDS、3′UTR/UTR区域、相对转录本位置和转录本长度。"
    "这种设计使模型学习的是同一坐标体系下的位点级m1A/m6A概率，而不是受基因组版本、窗口中心偏移或区域分布差异影响的混合信号。"
)

INTRO_SEQUENCE_DETAIL = (
    "StructDualRM的序列分支采用单核苷酸级编码，而不是将AAUCGGA一类序列先切成固定k-mer词表。"
    "每个位置的A/C/G/U被映射为可训练的base embedding，并与绝对位置embedding和距中心A的距离embedding相加后输入Transformer。"
    "随后，模型同时提取中心A隐藏状态、中心附近11 nt局部均值和中心感知注意力池化结果，因此既保留修饰位点本身的信息，也保留其左右局部序列环境。"
)

INTRO_STRUCTURE_DETAIL = (
    "结构分支则把RNAfold输出拆成两类互补信息。第一类是标量结构特征，包括中心碱基是否配对、中心11 nt和21 nt配对比例、完整101 nt配对比例、单位长度最低自由能以及中心附近环跨度；"
    "这些特征经标准化后进入结构MLP。第二类是由点括号结构解析得到的逐位点配对标志和碱基配对伙伴索引，用于构造成对感知图分支：相邻核苷酸提供RNA骨架边，RNAfold预测的配对碱基提供长程结构边。"
    "最终，序列Transformer、pair-aware graph、RNAfold标量结构和转录本上下文在标签特异性门控模块中融合，分别形成m1A和m6A的预测表征并输出两个概率。"
)

SEQ_ENCODING = (
    "核苷酸被编码为整数标记，并保留一个填充/未知标记。这里的输入不是k-mer词袋或固定长度k-mer token，而是101 nt窗口内逐位点的单碱基token；"
    "这种编码可以让模型直接学习中心A两侧每个位置的上下文，同时避免把不同位置的相同k-mer混为同一特征。"
    "对于101核苷酸窗口中位置j，序列编码器使用核苷酸、绝对位置和中心距离嵌入之和："
)

SEQ_POOLING = (
    "其中，L = 101，h_ij为位置j处的Transformer输出，[·; ·]表示向量拼接。最终序列表征拼接三部分：中心A处的隐藏状态、中心周围11核苷酸的局部均值，以及由中心隐藏状态引导的注意力池化向量。"
    "区域上下文由5′UTR、CDS、3′UTR或UTR标签的嵌入表示；数值上下文包括相对转录本位置和log10转录本长度。"
    "因此，模型既能识别局部motif，也能区分同一motif出现在不同转录本区域和不同相对位置时的含义。"
)

STRUCTURE_REPRESENTATION = (
    "RNA二级结构使用RNAfold在与序列建模相同的101核苷酸窗口上估计。RNAfold的点括号输出首先被转换为两类逐位点编码："
    "一是paired flags，表示每个位置是否参与碱基配对；二是pair partner index，记录每个配对位置的伙伴坐标，未配对位置记为缺失并在图层中掩蔽。"
    "此外，RNAfold输出被概括为标量结构向量："
)

STRUCTURE_SCALARS = (
    "其中，χ_i^(0)表示中心碱基是否配对，ρ_i^(11)、ρ_i^(21)和ρ_i^(101)分别为中心11核苷酸、中心21核苷酸和完整101核苷酸窗口中的配对比例，MFE_i/L为按序列长度归一化的最低自由能，ℓ_i为中心周围的环跨度。"
    "这些标量结构特征在输入模型前仅使用训练集均值和标准差进行标准化，以避免验证集和测试集信息泄漏。"
)

ARCHITECTURE_TEXT = (
    "StructDualRM结合四个证据流：序列表征z_i^seq、成对感知图表征z_i^graph、标量RNAfold表征z_i^struct，以及转录本上下文z_i^ctx。"
    "其中z_i^seq来自Transformer后的中心状态、局部均值和中心感知注意力；z_i^graph来自两层结构图更新后的中心状态、中心±10 nt局部均值、图注意力池化和中心碱基配对伙伴表征；"
    "z_i^struct来自6维RNAfold标量特征的MLP；z_i^ctx来自转录本区域嵌入、相对转录本位置和log10转录本长度。"
    "在完整M4模型中，标签特异性门控不是把m1A和m6A预测结果彼此作为输入，而是分别为m1A和m6A学习一个结构证据权重，用来调制图结构特征和标量结构特征："
)

GATE_FORMULA = (
    "γ_ic = σ(W_{γ,c} [z_i^seq; z_i^graph; z_i^struct; z_i^ctx] + b_{γ,c}), (6)"
)

GATE_EXPLANATION = (
    "其中，c ∈ {m1A, m6A}，γ_ic为0到1之间的标签特异性门控值。门控后的标签特异性表征为："
)

HEAD_TEXT = (
    "随后，m1A和m6A各自使用独立预测头h_c(·)产生logit o_ic，并通过sigmoid函数转换为概率："
)

PROB_FORMULA = (
    "o_ic = h_c(z_ic),  p_ic = σ(o_ic) = 1 / (1 + exp(-o_ic)). (8)"
)

GATE_DETAIL = (
    "这种结构使两个任务共享同一套序列、结构和上下文编码器，同时允许不同修饰对结构证据的依赖程度不同。"
    "在生物学上，这一点尤其重要：m1A可能更直接影响Watson-Crick配对界面，而m6A更多体现序列基序、局部可及性和reader相关环境；统一编码加标签特异性门控可以在共享信息和修饰特异性之间取得平衡。"
)


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(t.text or "" for t in paragraph.findall(".//w:t", NS))


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    text_elems = paragraph.findall(".//w:t", NS)
    if not text_elems:
        return
    text_elems[0].text = text
    for elem in text_elems[1:]:
        elem.text = ""


def make_like(template: ET.Element, text: str) -> ET.Element:
    new_paragraph = copy.deepcopy(template)
    set_paragraph_text(new_paragraph, text)
    return new_paragraph


def find_paragraph(paragraphs: list[ET.Element], prefix: str) -> int:
    for idx, paragraph in enumerate(paragraphs):
        if paragraph_text(paragraph).startswith(prefix):
            return idx
    raise RuntimeError(f"Could not find paragraph starting with: {prefix}")


def patch_document(xml_bytes: bytes) -> bytes:
    namespaces: list[tuple[str, str]] = []
    for _, ns_decl in ET.iterparse(BytesIO(xml_bytes), events=("start-ns",)):
        if ns_decl not in namespaces:
            namespaces.append(ns_decl)
    for prefix, uri in namespaces:
        ET.register_namespace(prefix, uri)

    root = ET.fromstring(xml_bytes)
    body = root.find(".//w:body", NS)
    if body is None:
        raise RuntimeError("word/document.xml has no w:body")

    paragraphs = body.findall("w:p", NS)

    intro_idx = find_paragraph(paragraphs, "在此，我们提出StructDualRM")
    intro_template = paragraphs[intro_idx]
    set_paragraph_text(intro_template, INTRO_FRAMEWORK)
    body.insert(intro_idx + 1, make_like(intro_template, INTRO_SEQUENCE_DETAIL))
    body.insert(intro_idx + 2, make_like(intro_template, INTRO_STRUCTURE_DETAIL))

    paragraphs = body.findall("w:p", NS)
    replacements = {
        "核苷酸被编码为整数标记": SEQ_ENCODING,
        "其中，L = 101": SEQ_POOLING,
        "RNA二级结构使用RNAfold": STRUCTURE_REPRESENTATION,
        "其中，χ_i^(0)": STRUCTURE_SCALARS,
        "StructDualRM结合四个证据流": ARCHITECTURE_TEXT,
        "γ_ic = σ": GATE_FORMULA,
        "其中，c ∈": GATE_EXPLANATION,
        "随后，独立的预测头": HEAD_TEXT,
        "p_ic = σ": PROB_FORMULA,
    }
    for prefix, new_text in replacements.items():
        idx = find_paragraph(paragraphs, prefix)
        set_paragraph_text(paragraphs[idx], new_text)

    paragraphs = body.findall("w:p", NS)
    z_idx = find_paragraph(paragraphs, "z_ic = [z_i^seq")
    body.insert(z_idx + 1, make_like(paragraphs[z_idx], GATE_DETAIL))

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not INPUT_DOCX.exists():
        raise FileNotFoundError(INPUT_DOCX)

    shutil.copy2(INPUT_DOCX, BACKUP_DOCX)

    with zipfile.ZipFile(INPUT_DOCX, "r") as zin:
        original_xml = zin.read("word/document.xml")
        patched_xml = patch_document(original_xml)
        with zipfile.ZipFile(OUTPUT_DOCX, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "word/document.xml":
                    data = patched_xml
                zout.writestr(info, data)

    with zipfile.ZipFile(OUTPUT_DOCX, "r") as check:
        check.testzip()
        xml = check.read("word/document.xml").decode("utf-8")
    required_phrases = [
        "单核苷酸级编码",
        "距中心A的距离embedding",
        "pair partner index",
        "标签特异性门控不是把m1A和m6A预测结果彼此作为输入",
        "o_ic = h_c(z_ic)",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in xml]
    if missing:
        raise RuntimeError(f"Patched document is missing required phrases: {missing}")

    print(f"backup={BACKUP_DOCX}")
    print(f"output={OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
