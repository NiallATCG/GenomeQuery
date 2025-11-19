#!/usr/bin/env python3
"""
Streamlit app: Genome Scan SNP Explorer (improved)

Improvements over previous version:
- Safer handling of uploaded files (Streamlit UploadedFile objects)
- Fallback messages when cyvcf2 is not installed
- Indexing of VCF by rsID for O(1) genotype lookup per sample
- Robust handling of missing alleles and phased genotypes
- Correct gamete generation and child genotype probability computation
- Clearer trait summary logic and graceful "Unknown" responses when data missing
- Minor UI/UX improvements and better error messages
"""
import streamlit as st
from collections import Counter
import gdown
import tempfile
import os

try:
    from cyvcf2 import VCF
except Exception:
    VCF = None

st.set_page_config(page_title="Genome Scan SNP Explorer", layout="wide")

# ── Trait Dictionary (same core content) ──
traits_info = {
    "Freckles": {"gene": "MC1R", "snps": ["rs1805007", "rs1805008"], "inheritance": "dominant", "description": "MC1R variants increase freckling"},
    "Hair Colour": {"gene": "MC1R", "snps": ["rs1805007", "rs1805008"], "inheritance": "dominant", "description": "MC1R variants shift pigment toward red"},
    "Eye Colour": {"gene": "HERC2", "snps": ["rs12913832"], "inheritance": "recessive", "description": "Blue vs brown eye colour"},
    "Skin Tone": {"gene": "SLC24A5", "snps": ["rs1426654"], "inheritance": "recessive", "description": "A/A lighter skin"},
    "Sprint Gene": {"gene": "ACTN3", "snps": ["rs1815739"], "inheritance": "dominant", "description": "Fast‑twitch muscle performance"},
    "Alcohol Flush": {"gene": "ALDH2", "snps": ["rs671"], "inheritance": "dominant", "description": "Asian flush reaction"},
    "Earwax Type": {"gene": "ABCC11", "snps": ["rs17822931"], "inheritance": "dominant", "description": "Wet vs dry earwax"},
    "Lactose Intolerance": {"gene": "MCM6", "snps": ["rs4988235"], "inheritance": "dominant", "description": "T allele maintains lactase"},
    "PTC Tasting": {"gene": "TAS2R38", "snps": ["rs713598", "rs1726866"], "inheritance": "dominant", "description": "PAV haplotype = taster"},
    "Coriander Taste": {"gene": "OR6A2", "snps": ["rs72921001"], "inheritance": "dominant", "description": "Soapy taste perception"},
    # Pharmacogenetics (examples)
    "Warfarin Response": {"gene": "VKORC1; CYP2C9; CYP4F2", "snps": ["rs9923231", "CYP2C9*2", "CYP2C9*3", "rs2108622"], "inheritance": "pharmacogenetic", "description": "Variants alter warfarin dose"},
    "Statin Myopathy": {"gene": "SLCO1B1; ABCG2", "snps": ["rs4149056", "rs2231142"], "inheritance": "pharmacogenetic", "description": "Higher statin exposure risk"},
    "Clopidogrel Response": {"gene": "CYP2C19", "snps": ["CYP2C19*2", "CYP2C19*3", "CYP2C19*17"], "inheritance": "pharmacogenetic", "description": "Activation differences"},
    "Opioid Response": {"gene": "CYP2D6", "snps": ["CYP2D6*3", "CYP2D6*4", "CYP2D6*5", "CYP2D6*6"], "inheritance": "pharmacogenetic", "description": "Poor vs ultrarapid metabolism"},
}

# ── Helpers ──
def download_vcf(url):
    """Download a file using gdown to a temp file and return path."""
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf")
    out.close()
    gdown.download(url, out.name, quiet=True)
    return out.name

def save_uploaded_file(uploaded_file):
    """Save a Streamlit UploadedFile to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1] or ".vcf")
    with open(tmp.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp.name

def build_sample_index(vcf_obj, sample):
    """
    Build a dict mapping rsID -> genotype for a given sample.
    Genotype is returned as a tuple: (allele1, allele2) where allele can be 0,1 or None (missing).
    """
    idx = vcf_obj.samples.index(sample)
    index = {}
    for rec in vcf_obj:
        rid = rec.ID
        if not rid:
            continue
        g = rec.genotypes[idx][:2]  # cyvcf2 returns [a1, a2, phased]
        # Normalize None-like values to None, integers remain
        a1 = None if g[0] is None or g[0] == -1 else int(g[0])
        a2 = None if g[1] is None or g[1] == -1 else int(g[1])
        index[rid] = (a1, a2)
    return index

def get_genotype_from_index(index, rsid):
    """Return genotype tuple (a1, a2) or None if not found."""
    return index.get(rsid)

def genotype_alt_count(genotypes):
    """
    Count the number of alternate alleles (1) across a list of genotype tuples.
    Skip missing data.
    """
    total = 0
    for gt in genotypes:
        if not gt:
            continue
        a1, a2 = gt
        if a1 == 1:
            total += 1
        if a2 == 1:
            total += 1
    return total

def get_gametes_from_gt(gt):
    """
    Given a genotype (a1, a2) return list of possible gamete alleles.
    Examples:
      (0,0) -> [0]
      (0,1) -> [0,1]
      (1,1) -> [1]
    If missing (None present), return empty list.
    """
    if not gt:
        return []
    a1, a2 = gt
    if a1 is None or a2 is None:
        return []
    if a1 == a2:
        return [a1]
    return [a1, a2]

def child_probs_from_parents(m_gt, f_gt):
    """
    Given mother and father genotypes (a1,a2), compute predicted child genotype probabilities as percentages.
    Returns dict mapping genotype string '0/0','0/1','1/1' to percentage.
    """
    m_gametes = get_gametes_from_gt(m_gt)
    f_gametes = get_gametes_from_gt(f_gt)
    if not m_gametes or not f_gametes:
        return None  # insufficient data

    combos = []
    for mg in m_gametes:
        for fg in f_gametes:
            # child genotype alleles (sorted to present 0/1 same as 1/0)
            child = tuple(sorted((mg, fg)))
            combos.append(child)
    counts = Counter(combos)
    total = sum(counts.values())
    probs = {}
    for geno, cnt in counts.items():
        if geno == (0, 0):
            name = "0/0"
        elif geno == (0, 1):
            name = "0/1"
        elif geno == (1, 1):
            name = "1/1"
        else:
            name = f"{geno[0]}/{geno[1]}"
        probs[name] = round(cnt / total * 100, 1)
    # Ensure all three common genotypes present
    for expected in ("0/0", "0/1", "1/1"):
        if expected not in probs:
            probs[expected] = 0.0
    return probs

def trait_summary(trait, info, index=None):
    """
    Create a human-readable summary for a trait using an index of rsID->genotype.
    If index is None or genotype missing, returns a conservative message.
    """
    if not index:
        return f"{info['description']} (no genotype data available)"

    gts = [get_genotype_from_index(index, s) for s in info["snps"]]
    # If all SNPs missing, say Unknown
    if all(gt is None for gt in gts):
        return f"{info['description']} (genotype unknown)"

    alt_count = genotype_alt_count(gts)

    # Trait-specific heuristics (kept simple and conservative)
    if trait == "Freckles":
        # If multiple alternate alleles across the MC1R snps, more pronounced
        if alt_count >= 3:
            return "Pronounced"
        if alt_count >= 1:
            return "Mild"
        return "None"
    if trait == "Eye Colour":
        gt = gts[0]
        if not gt:
            return "Unknown"
        return "Blue" if gt == (0, 0) else "Brown"
    if trait == "Hair Colour":
        if alt_count >= 2:
            return "Red"
        if alt_count == 1:
            return "Auburn / mixed"
        return "Non-red"
    if trait == "Skin Tone":
        gt = gts[0]
        if not gt:
            return "Unknown"
        if gt == (1, 1):
            return "Light"
        if 1 in gt:
            return "Intermediate"
        return "Dark"
    if trait == "Sprint Gene":
        gt = gts[0]
        if not gt:
            return "Unknown"
        # rs1815739 (ACTN3): 0 = R (functional), 1 = X (stop)
        return "Reduced sprint" if gt == (1, 1) else "No major reduction"
    # Default fallback
    return f"{info['description']} (partial genotype available)"

# ── UI ──
st.title("Genome Scan SNP Explorer (improved)")

mode = st.sidebar.radio("Mode", ["Individual", "Child Predictor"])

if VCF is None:
    st.sidebar.error("cyvcf2 is not available in the environment. Install it (`pip install cyvcf2`) to enable VCF parsing.")
    st.sidebar.info("You can still paste a Google Drive link to a VCF; the app will attempt to download, but parsing requires cyvcf2.")

if mode == "Individual":
    method = st.sidebar.radio("Upload method", ["Local", "Google Drive", "Demo"])
    vcf_path = None
    tmp_files = []
    try:
        if method == "Local":
            uploaded = st.sidebar.file_uploader("Upload VCF", type=["vcf", "vcf.gz"])
            if uploaded is not None:
                vcf_path = save_uploaded_file(uploaded)
                tmp_files.append(vcf_path)
        elif method == "Google Drive":
            url = st.sidebar.text_input("Google Drive link (or direct download link)")
            if url:
                with st.spinner("Downloading..."):
                    vcf_path = download_vcf(url)
                    tmp_files.append(vcf_path)
        else:  # Demo not implemented with actual file; show descriptions only
            st.sidebar.info("Demo mode: show trait definitions without parsing a VCF.")
            vcf_path = None

        if vcf_path and VCF:
            vcf = VCF(vcf_path)
            sample = st.sidebar.selectbox("Sample", vcf.samples)
            st.subheader("Trait Summaries")
            index = build_sample_index(vcf, sample)
            for trait, info in traits_info.items():
                st.write(f"{trait}: {trait_summary(trait, info, index)}")
        else:
            if method != "Demo":
                st.info("No VCF parsed; install cyvcf2 and upload a VCF to see genotype-derived summaries.")
            st.subheader("Trait Definitions")
            for trait, info in traits_info.items():
                st.write(f"{trait}: {info['description']} (gene: {info['gene']})")
    finally:
        # clean up temporary files
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass

else:  # Child Predictor
    st.sidebar.info("Upload one VCF for each parent. The app will attempt to find the first SNP listed for each trait and predict child genotype percentages.")
    mom_file = st.sidebar.file_uploader("Mother VCF", type=["vcf", "vcf.gz"])
    dad_file = st.sidebar.file_uploader("Father VCF", type=["vcf", "vcf.gz"])
    tmp_files = []
    try:
        mom_path = save_uploaded_file(mom_file) if mom_file else None
        dad_path = save_uploaded_file(dad_file) if dad_file else None
        if mom_path:
            tmp_files.append(mom_path)
        if dad_path:
            tmp_files.append(dad_path)

        if mom_path and dad_path and VCF:
            vcf_m = VCF(mom_path)
            vcf_f = VCF(dad_path)
            mom = st.sidebar.selectbox("Mother sample", vcf_m.samples)
            dad = st.sidebar.selectbox("Father sample", vcf_f.samples)
            st.subheader("Predicted Child Genotypes")
            idx_m = build_sample_index(vcf_m, mom)
            idx_f = build_sample_index(vcf_f, dad)
            for trait, info in traits_info.items():
                # pick first SNP to demonstrate
                if not info["snps"]:
                    continue
                snp = info["snps"][0]
                m_gt = get_genotype_from_index(idx_m, snp)
                f_gt = get_genotype_from_index(idx_f, snp)
                if not m_gt or not f_gt:
                    st.write(f"{trait} ({snp}): Insufficient genotype data")
                    continue
                probs = child_probs_from_parents(m_gt, f_gt)
                if probs is None:
                    st.write(f"{trait} ({snp}): Insufficient genotype data")
                else:
                    st.write(f"{trait} ({snp}): {probs}")
        else:
            st.info("Upload both parent VCFs and ensure cyvcf2 is installed to compute child genotype probabilities.")
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass
