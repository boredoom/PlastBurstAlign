import re
import os
from collections import OrderedDict, defaultdict
from re import compile as re_compile
from re import findall, match, sub
from typing import Any, ClassVar, Optional

from Bio import SeqUtils
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .extraction_report import ExtractionReport
from .logging_ops import logger as log
from .qc import (
    ERR_COORDINATE_OVERRUN,
    ERR_TOPOLOGY_INVERSION,
    CdsCodingValidator,
    QuarantineLedger,
    circular_intron_bounds,
    feature_span_ok,
    location_raw_coords,
    record_topology,
)
from .storage import DiskBackedFeatureStore, FeatureStoreView
from .user_parameters import SelectMode

# Anything outside the IUPAC nucleotide alphabet cannot be aligned: such a feature
# is skipped (and reported) instead of poisoning its region.
_INVALID_RESIDUES = re_compile(r"[^ACGTURYSWKMBDHVNacgturyswkmbdhvn]")
# Region names become file names; real gene names are far shorter.
MAX_REGION_NAME_CHARS = 100

# ==============================================================================
# UPSTREAM PLASTID GENE CURATION DICTIONARIES
# ==============================================================================

# Foreign Contaminants: Assembly leaks and transgenic selection markers
CONTAMINANT_BLOCKLIST = frozenset({
    # Plant/Animal Mitochondrial Genes
    "cox1", "cox2", "cox3", "cob", "atp6", "atp9",
    "nad1", "nad2", "nad3", "nad4", "nad4l", "nad5", "nad6", "nad7", "nad9",
    # Transgenic Selection Vectors
    "aada", "dfr", "bla", "neo", "hpt", "nptii", "gfp", "uida", "bar", "pat"
})

# Author Spelling Variants & Aliases -> Canonical CDS
CANONICAL_CDS_ALIASES = {
    # rbcL Aliases & Typos
    "ribulose": "rbcL", "ribulosebisphosphatecarboxylase": "rbcL",
    "ribulose15bisphosphatecarboxylaselargesubunit": "rbcL",
    "rbcl2": "rbcL", "rbcl29": "rbcL",

    # RNA Polymerase Operon
    "rpobb": "rpoB", "rpoba": "rpoB", "rpo": "rpoB",
    "rpoc1a": "rpoC1", "rpoc1b": "rpoC1",
    "rpoc2a": "rpoC2", "rpoc2b": "rpoC2",

    # Ribosomal Protein Typos & Punctuation
    "rpl336": "rpl36", "rpi36": "rpl36", "rpl_36": "rpl36",
    "rpl_32": "rpl32", "rpl_5": "rpl5",
    "rp12": "rpl2", "rpi2": "rpl2",
    "ros12": "rps12", "rsp12": "rps12",
    "rps_14": "rps14", "rs14": "rps14",
    "rps_16": "rps16", "rps_19": "rps19",
    "rps2a": "rps2", "rps2b": "rps2",
    "rps15a": "rps15", "rps15b": "rps15",

    # Photosystem & ATP Synthase Aliases
    "maturase": "matK", "matuase": "matK", "mat_k": "matK", "mat": "matK",
    "psba1": "psbA", "psba2": "psbA", "psba": "psbA",
    "psbd1": "psbD",
    "pbn": "psbN", "pbt": "psbT",
    "psi": "psaI", "psj": "psaJ", "psaii": "psaB",
    "atp_b": "atpB", "atp_e": "atpE", "atp_f": "atpF",
    "ndhb1": "ndhB", "ndhb2": "ndhB",
    "ndhndhb": "ndhB", "ndhndhh": "ndhH",

    # Split Inverted Repeat Boundaries
    "ycf1a": "ycf1", "ycf1b": "ycf1",
    "ycf2c": "ycf2", "ycf2n": "ycf2",
}

class PlastidData:
    mode: SelectMode
    files: list[str]
    num_taxa: int
    order: str
    store: DiskBackedFeatureStore
    nucleotides: FeatureStoreView
    proteins: FeatureStoreView | None
    order_map: dict[str, int]
    extraction_report: ExtractionReport

    def __init__(self, user_params: dict[str, Any]):
        """
        Data structure to hold the extracted annotations.
        """
        self.user_params = user_params
        self._set_mode(user_params)
        self._set_files(user_params)
        self._set_order_fun(user_params)
        # filled by `ExtractAndCollect.extract()`; empty until then
        self.extraction_report = ExtractionReport()

        db_path = os.path.join(user_params.get("out_dir", "."), "feature_store.db")
        self.store = DiskBackedFeatureStore(db_path)
        # Main-process quarantine ledger; workers return records via IPC payloads.
        self.qc_ledger = QuarantineLedger()

        self._set_nucleotides()
        self._set_proteins()

    def _set_mode(self, user_params: dict[str, Any]):
        self.mode = user_params.get("select_mode")  # type: ignore[assignment]

    def _set_nucleotides(self):
        self.nucleotides = FeatureStoreView(self.store, "nucl")

    def _set_proteins(self):
        self.proteins = (
            FeatureStoreView(self.store, "prot")
            if self.mode.collects_proteins()
            else None
        )

    def _set_files(self, user_params: dict[str, Any]):
        """Collects the input files in a deterministic (sorted by name) order.

        `os.listdir` order is filesystem dependent; the position of a file in this
        list becomes the `file_index` of its records and thereby the row order of
        every alignment and of the concatenated supermatrix.

        Cluster worker and finalizer runs operate on the existing feature store
        only, so a missing input directory is tolerated there.
        """
        in_dir = user_params.get("in_dir")
        is_cluster_worker_or_final = (
            user_params.get("align_partition") is not None
            or user_params.get("finalize_slurm") is True
        )
        try:
            names = os.listdir(in_dir)
        except OSError:
            if not is_cluster_worker_or_final:
                raise
            names = []
        fileext = user_params.get("fileext")
        self.files = [
            os.path.join(in_dir, f) for f in sorted(names) if f.endswith(fileext)
        ]
        self.num_taxa = len(self.files)

    def _set_order_fun(self, user_params: dict[str, Any]):
        self.order = user_params.get("order")

    def add_nucleotides(
        self, pdict: Any, inv_map: dict[str, str] | None = None
    ) -> None:
        """Merges extracted dictionary into the disk store."""
        if not pdict:
            return
        if inv_map is None:
            is_igs = isinstance(pdict, IntergenicDict)
            inv_map = getattr(pdict, "feat_name_inv_map", {}) if is_igs else None

        with self.store.transaction():
            self.store.add_pdict(pdict, seq_type="nucl", inv_map=inv_map)

    def add_proteins(self, pdict: Any) -> None:
        """Merges extracted proteins into the disk store."""
        if not pdict or not self.mode.collects_proteins():
            return
        with self.store.transaction():
            self.store.add_pdict(pdict, seq_type="prot", inv_map=None)

    def set_order_map(self):
        order_list = self.store.get_active_regions(self.order)
        self.order_map = {nuc: index for index, nuc in enumerate(order_list)}

    def remove_nuc(self, feat_name: str):
        self.store.remove_region(feat_name)


class PlastidDict(OrderedDict):
    def __init__(self):
        """
        A dictionary used to store extracted sequence annotations.
        """
        super().__init__()
        self.plastome_feats = defaultdict(set)

    def _not_id(self, feat_name: str, rec_name: str) -> bool:
        rec_set = self.plastome_feats.get(rec_name)
        return True if not rec_set else feat_name not in rec_set

    def _is_feat(self, feat_name: str) -> bool:
        return feat_name in self.keys()

    def add_feature(self, feature: "PlastidFeature"):
        """
        Adds an annotation to the dictionary. If an annotation with the same name has already been added
        for the plastome (record name), it will not be added.

        Args:
            feature: A plastid feature annotation.
        """
        if feature.seq_obj is None:
            feature.log_exception()
            return

        is_feat = self._is_feat(feature.feat_name)
        not_id = self._not_id(feature.feat_name, feature.rec_name)
        # if nuc list exists, and this nuc has not been added for the plastome
        if is_feat and not_id:
            self[feature.feat_name].append(feature.get_record())
        # if the nuc list does not exist
        elif not is_feat:
            self[feature.feat_name] = [feature.get_record()]
        # record that the feature for the plastome has been added
        self.plastome_feats[feature.rec_name].add(feature.feat_name)


class IntergenicDict(PlastidDict):
    def __init__(self):
        """
        A dictionary used to store extracted intergenic regions.
        """
        super().__init__()
        self.feat_name_inv_map = {}

    def add_feature(self, igs: "IntergenicFeature"):
        super().add_feature(igs)
        self.feat_name_inv_map[igs.feat_name] = igs.inv_feat_name


# -----------------------------------------------------------------#


class GeneNameCleaner:
    _rrna_gene_names: ClassVar[dict[str, str]] = {
        "16s": "rrn16",
        "16srrna": "rrn16",
        "16sribosomalrna": "rrn16",
        "rrna16": "rrn16",
        "rrna16s": "rrn16",
        "rns": "rrn16",
        "ssu": "rrn16",
        "ssurrna": "rrn16",
        "ssurdna": "rrn16",
        "rrn16": "rrn16",
        "rrn16s": "rrn16",
        "23s": "rrn23",
        "23srrna": "rrn23",
        "23sribosomalrna": "rrn23",
        "rrna23": "rrn23",
        "rrna23s": "rrn23",
        "rnl": "rrn23",
        "lsu": "rrn23",
        "lsurrna": "rrn23",
        "lsurdna": "rrn23",
        "rrn23": "rrn23",
        "rrn23s": "rrn23",
        "4.5s": "rrn4.5",
        "4.5srrna": "rrn4.5",
        "4.5sribosomalrna": "rrn4.5",
        "rrna4.5": "rrn4.5",
        "rrna4.5s": "rrn4.5",
        "rrn45": "rrn4.5",
        "rrn45s": "rrn4.5",
        "rrn4.5": "rrn4.5",
        "rrn4.5s": "rrn4.5",
        "5s": "rrn5",
        "5srrna": "rrn5",
        "5sribosomalrna": "rrn5",
        "rrna5": "rrn5",
        "rrna5s": "rrn5",
        "5srdna": "rrn5",
        "rrn5": "rrn5",
        "rrn5s": "rrn5",
    }

    def __init__(self, table_id: int = 11):
        """
        Extracts gene names from `SeqFeature` objects and standardizes gene names.
        tRNA names use amino acid abbreviations and a specified codon table to add
        anticodon information. The four plastid rRNA names are standardized to
        `rrn16`, `rrn23`, `rrn4.5`, and `rrn5`.
        The default codon table used is INSDC table 11.

        Args:
            table_id: Genetic code table ID to use for mapping from amino acids to codons.
                For more, reference https://www.insdc.org/submitting-standards/genetic-code-tables/.
        """
        self.codon_table = CodonTable.unambiguous_dna_by_id[table_id]

    def _amino3_to_anticodon(self, amino_code: str) -> Optional[str]:
        if len(amino_code) != 3:
            return None
        amino_char = SeqUtils.seq1(amino_code)
        codon = self.codon_table.back_table.get(amino_char)
        if codon:
            anticodon = Seq(codon).reverse_complement_rna()
            return str(anticodon)
        else:
            return None

    @staticmethod
    def _parse_product_to_gene(product: str) -> Optional[str]:
        """Map canonical GenBank /product annotations to standardized plastid gene names."""
        if not product:
            return None
        s = product.strip()
        if s.lower().startswith("product:"):
            s = s[8:].strip()

        # Handle direct typo Psal for psaI
        if s.lower() == "psal":
            return "psaI"

        # Historical Reading Frame aliases (RF1 -> ycf1, RF21 -> ycf2, RF34 -> ycf3)
        m_rf = re.search(r"(?i)(?:putative|hypothetical)?\s*chloroplast\s*rf(\d+)\b", s)
        if m_rf:
            rf_map = {"1": "ycf1", "21": "ycf2", "34": "ycf3"}
            if m_rf.group(1) in rf_map:
                return rf_map[m_rf.group(1)]

        # Direct short symbols (e.g., PsbA, rbcL, Ycf1, lhbA, ClpP, MatK)
        m_sym = re.match(r"^([a-zA-Z]{3,4}\d{0,2}[a-zA-Z]?)$", s)
        if m_sym:
            sym = m_sym.group(1)
            return sym[0].lower() + sym[1:]

        # Ribosomal proteins: "ribosomal protein S12" -> "rps12"
        m_rp = re.match(r"(?i)ribosomal protein ([sl])(\d+)", s)
        if m_rp:
            return f"rp{m_rp.group(1).lower()}{m_rp.group(2)}"

        # Photosystem II
        if re.search(r"(?i)photosystem ii (?:reaction center )?protein d1\b", s):
            return "psbA"
        if re.search(r"(?i)photosystem ii (?:reaction center )?protein d2\b", s):
            return "psbD"
        if re.search(r"(?i)photosystem ii (?:44\s*kda|cp43)", s):
            return "psbC"
        if re.search(r"(?i)photosystem ii 47\s*kda", s):
            return "psbB"
        if re.search(r"(?i)photosystem ii (?:43\s*kda|protein v\b)", s):
            return "psbC"
        if re.search(r"(?i)photosystem ii (?:phosphoprotein|protein h\b|subunit h\b)", s) or re.search(r"(?i)photosystem phosphoprotein", s):
            return "psbH"
        if re.search(r"(?i)photosystem ii cytochrome b559 alpha", s):
            return "psbE"
        if re.search(r"(?i)photosystem ii cytochrome b559 beta", s):
            return "psbF"
        m_psb = re.search(r"(?i)photosystem ii (?:protein|subunit) ([a-z0-9]+)", s)
        if m_psb:
            return f"psb{m_psb.group(1).upper()}"

        # Photosystem I
        if re.search(r"(?i)photosystem i p700 (?:chlorophyll a )?apoprotein a1", s):
            return "psaA"
        if re.search(r"(?i)photosystem i p700 (?:chlorophyll a )?apoprotein a2", s):
            return "psaB"
        if re.search(r"(?i)photosystem i subunit ix", s):
            return "psaI"
        if re.search(r"(?i)photosystem i subunit viii", s):
            return "psaD"
        if re.search(r"(?i)photosystem i subunit vii", s):
            return "psaC"
        m_ycf_psi = re.search(r"(?i)photosystem i assembly protein (ycf\d+)", s)
        if m_ycf_psi:
            return m_ycf_psi.group(1).lower()
        m_psa = re.search(r"(?i)photosystem i (?:protein|subunit) ([a-z0-9]+)", s)
        if m_psa:
            return f"psa{m_psa.group(1).upper()}"

        # Rubisco large subunit
        if re.search(r"(?i)ribulose.*large subunit", s) or re.search(r"(?i)ribulose-1,5-bisphosphate carboxylase/oxygenase$", s):
            return "rbcL"

        # Maturase K
        if re.search(r"(?i)maturase k", s):
            return "matK"

        # ATP synthase
        if re.search(r"(?i)atp (?:synthase|systhase).*cf1 alpha", s) or re.search(r"(?i)cf1 subunit alpha", s):
            return "atpA"
        if re.search(r"(?i)atp (?:synthase|systhase).*cf1 beta", s) or re.search(r"(?i)cf1 subunit beta", s):
            return "atpB"
        if re.search(r"(?i)atp (?:synthase|systhase).*cf1 epsilon", s) or re.search(r"(?i)cf1 subunit epsilon", s):
            return "atpE"
        if re.search(r"(?i)atp (?:synthase|systhase).*(?:cf0|cfo) b", s) or re.search(r"(?i)cf0 subunit ii\b", s):
            return "atpF"
        if re.search(r"(?i)atp (?:synthase|systhase).*(?:cf0|cfo) c", s) or re.search(r"(?i)cf0 subunit iii", s):
            return "atpH"
        if re.search(r"(?i)atp (?:synthase|systhase).*(?:cf0|cfo) (?:a|subunit i\b|subunit iv\b)", s) or re.search(r"(?i)cf0 subunit iv", s):
            return "atpI"

        # Cytochromes
        if re.search(r"(?i)cytochrome f\b", s):
            return "petA"
        if re.search(r"(?i)cytochrome b6\b", s):
            return "petB"
        if re.search(r"(?i)cytochrome b6/f complex subunit iv", s):
            return "petD"
        if re.search(r"(?i)cytochrome b6/f complex subunit v\b", s) or re.search(r"(?i)cytochrome b6/f subunit g", s):
            return "petG"
        if re.search(r"(?i)cytochrome b6/f complex subunit vi\b", s) or re.search(r"(?i)cytochrome b6/f subunit l", s):
            return "petL"
        if re.search(r"(?i)cytochrome b6/f complex subunit viii\b", s) or re.search(r"(?i)cytochrome b6/f subunit n", s):
            return "petN"
        if re.search(r"(?i)cytochrome b559 subunit alpha", s):
            return "psbE"
        if re.search(r"(?i)cytochrome b559 subunit beta", s):
            return "psbF"

        # NADH dehydrogenase / NDH
        m_ndh = re.search(r"(?i)(?:nadh dehydrogenase|nad\(p\)h-quinone oxidoreductase) subunit ([a-z0-9]+)", s)
        if m_ndh:
            sub = m_ndh.group(1).upper()
            mapping = {
                "1": "ndhA", "2": "ndhB", "3": "ndhC", "4": "ndhD", "4L": "ndhE",
                "5": "ndhF", "6": "ndhJ", "7": "ndhH",
                "A": "ndhA", "B": "ndhB", "C": "ndhC", "D": "ndhD", "E": "ndhE",
                "F": "ndhF", "G": "ndhG", "H": "ndhH", "I": "ndhI", "J": "ndhJ", "K": "ndhK"
            }
            return mapping.get(sub, f"ndh{sub}")

        # RNA polymerase: check beta'' (rpoC2) before beta' (rpoC1) before beta (rpoB)
        if re.search(r"(?i)rna polymerase (?:subunit )?alpha", s):
            return "rpoA"
        if re.search(r"(?i)rna polymerase.*beta[''’’]{2}", s) or re.search(r"(?i)rna polymerase.*beta''", s):
            return "rpoC2"
        if re.search(r"(?i)rna polymerase.*beta['’]", s) or re.search(r"(?i)rna polymerase.*beta'", s):
            return "rpoC1"
        if re.search(r"(?i)rna polymerase (?:subunit )?beta(?![a-zA-Z'’])", s):
            return "rpoB"

        # accD, clpP, ccsA, cemA, infA
        if re.search(r"(?i)acetyl-coa carboxylase", s):
            return "accD"
        if re.search(r"(?i)clp protease", s) or re.search(r"(?i)atp-dependent.*protease", s):
            return "clpP"
        if re.search(r"(?i)c-type cytochrome synthesis", s) or re.search(r"(?i)cytochrome c biogenesis", s):
            return "ccsA"
        if re.search(r"(?i)envelope membrane (?:carbon uptake )?protein", s):
            return "cemA"
        if re.search(r"(?i)initiation factor", s):
            return "infA"
        m_ycf = re.search(r"(?i)ycf(\d+)", s)
        if m_ycf:
            return f"ycf{m_ycf.group(1)}"

        return None

    @staticmethod
    def get_gene(feat: SeqFeature) -> Optional[str]:
        """
        Finds, normalizes, and validates the gene name for a plastid feature.
        Filters contaminants and maps aliases into canonical nomenclature.
        """
        raw_val = None
        if feat.qualifiers.get("gene"):
            raw_val = feat.qualifiers["gene"][0].strip()
        elif feat.qualifiers.get("product"):
            prod_val = feat.qualifiers["product"][0].strip()
            raw_val = GeneNameCleaner._parse_product_to_gene(prod_val)

        if not raw_val:
            return None

        # Clean punctuation and leading/trailing underscores
        clean_key = re.sub(r"[_\W]+", "", raw_val).lower()

        # 1. Foreign Contaminant Shield (Reject mitochondrial leaks and vectors)
        if clean_key in CONTAMINANT_BLOCKLIST:
            return None

        # 2. Canonical Alias Translation
        if clean_key in CANONICAL_CDS_ALIASES:
            return CANONICAL_CDS_ALIASES[clean_key]

        # 3. Strip non-alphanumeric noise while retaining standard nomenclature
        cleaned_name = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", raw_val)
        return cleaned_name if cleaned_name else None

    def clean_trna_gene(
        self, gene: Optional[str], cut_trna: bool = False
    ) -> Optional[str]:
        """
        Standardizes a tRNA gene name by case, delimiter usage, and anticodon.
        If there is no initial gene name (`None`), `None` will be returned.

        Args:
            gene: Gene name.
            cut_trna: Option to disregard additional qualifiers from the name (default is `False`).

        Returns:
            Cleaned gene name.

        """
        if not gene:
            return None

        # look for standardized gene name
        gene_pattern = r"^[a-zA-Z]{3}[a-zA-Z0-9]+"
        gene_sub = match(gene_pattern, gene)
        # if non-standard, just return a "safe" version of the name
        if not gene_sub:
            return sub(r"\W+", "_", gene.replace("-", "_"))
        # the first match is the beginning of the cleaned name
        cleaned_gene = gene_sub.group()[0:3].lower() + gene_sub.group()[3:].upper()
        if cut_trna:
            return cleaned_gene

        # look for tRNA qualifiers
        gene = gene[len(cleaned_gene) :]
        trna_pattern = r"([a-zA-Z]{3})"
        qual_subs = findall(trna_pattern, gene)
        # if not tRNA, return the gene
        if not qual_subs:
            return cleaned_gene

        # handle each tRNA qualifier appropriately
        codon_pattern = r"\b[ACGTUacgtu]{3}\b"
        anticodon = None
        while not anticodon and qual_subs:
            qualifier = qual_subs.pop()
            # qualifier is codon
            if match(codon_pattern, qualifier):
                anticodon = qualifier.upper()
            # qualifier is amino acid?
            else:
                anticodon = self._amino3_to_anticodon(qualifier)
        if anticodon:
            cleaned_gene += "_" + anticodon
        return cleaned_gene

    def clean_rrna_gene(self, gene: Optional[str]) -> Optional[str]:
        """Standardizes recognized plastid rRNA names to their canonical gene names."""
        if not gene:
            return None
        normalized_gene = sub(r"[^a-z0-9.]", "", gene.casefold())
        canonical_name = self._rrna_gene_names.get(normalized_gene)
        if canonical_name:
            return canonical_name
        return self.clean_gene(gene)

    def clean_gene(self, gene: Optional[str], cut_trna: bool = False) -> Optional[str]:
        """Standardizes a gene name using the legacy general/tRNA cleaning behavior."""
        return self.clean_trna_gene(gene, cut_trna)

    def get_safe_gene(self, feat: SeqFeature) -> Optional[str]:
        """
        Finds and returns the cleaned gene name for a feature. If the feature is not a gene, `None` will be returned.

        Args:
            feat: `Biopython` feature.

        Returns:
            Cleaned gene name.

        """
        gene = self.get_gene(feat)
        if feat.type == "tRNA":
            return self.clean_trna_gene(gene)
        if feat.type == "rRNA":
            return self.clean_rrna_gene(gene)
        if gene and gene.casefold().startswith("trn"):
            return self.clean_trna_gene(gene)
        if gene and sub(r"[^a-z0-9.]", "", gene.casefold()) in self._rrna_gene_names:
            return self.clean_rrna_gene(gene)
        return self.clean_gene(gene)


# -----------------------------------------------------------------#


class PlastidFeature:
    name_cleaner = GeneNameCleaner()

    _type: str = "plastid feature"
    _default_exception: str = "exception"
    _log_fun = log.warning

    def __init__(self, record: SeqRecord, feature: SeqFeature):
        """
        Primarily a wrapper for a `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        Args:
            record: The record that the feature is being extracted from.
            feature: The feature to be extracted.
        """
        self._exception = None
        self.seq_obj = None
        self._qc_reason: str | None = None
        self._qc_coords: str = ""
        self._qc_observed_len: int = 0
        self._parent_len = (
            len(record.seq) if record is not None and record.seq is not None else 0
        )
        self._topology = record_topology(record)

        self._set_feat_name(feature)
        self._set_rec_name(record)
        self._set_feature(feature)
        self._set_seq_obj(record)
        self._set_seq_name()

    def _set_feat_name(self, feature: SeqFeature):
        self.feat_name = self.name_cleaner.get_safe_gene(feature)

    def _set_rec_name(self, record: SeqRecord):
        self.rec_name = record.name

    def _set_feature(self, feature: SeqFeature):
        self.feature = feature

    def _reject_qc(self, reason: str, coords: str, observed_len: int) -> None:
        """Mark this feature as Tier-1 quarantined (sequence discarded)."""
        self.seq_obj = None
        self._qc_reason = reason
        self._qc_coords = coords
        self._qc_observed_len = int(observed_len)
        self._set_exception(ValueError(reason))

    def _location_for_span(self) -> Any:
        """Return the Biopython location object used for span checks."""
        feat = self.feature
        if feat is None:
            return None
        return getattr(feat, "location", feat)

    def _set_seq_obj(self, record: SeqRecord):
        if self.feature is None:
            if self._qc_reason is None:
                self._reject_qc(ERR_TOPOLOGY_INVERSION, "", 0)
            else:
                self.seq_obj = None
                self._set_exception(ValueError(self._qc_reason))
            return
        parent_len = self._parent_len or (
            len(record.seq) if record is not None and record.seq is not None else 0
        )
        topology = self._topology or record_topology(record)
        loc = self._location_for_span()
        ok, reason, observed = feature_span_ok(loc, parent_len, topology=topology)
        if not ok and reason is not None:
            self._reject_qc(reason, location_raw_coords(loc), observed)
            return
        try:
            seq = self.feature.extract(record).seq
            bad = _INVALID_RESIDUES.search(str(seq))
            if bad:
                raise ValueError(
                    f"invalid nucleotide character {bad.group()!r} at position"
                    f" {bad.start() + 1} of the feature"
                )
            if parent_len > 0 and len(seq) > parent_len:
                self._reject_qc(
                    ERR_COORDINATE_OVERRUN,
                    location_raw_coords(loc),
                    len(seq),
                )
                return
            self.seq_obj = seq
        except Exception as e:
            self._set_exception(e)

    def _set_seq_name(self):
        self.seq_name = f"{self.feat_name}_{self.rec_name}"
        if (
            isinstance(self.feat_name, str)
            and len(self.feat_name) > MAX_REGION_NAME_CHARS
        ):
            self.seq_obj = None
            self._set_exception(
                ValueError(
                    f"region name of {len(self.feat_name)} characters exceeds the"
                    f" {MAX_REGION_NAME_CHARS} character limit"
                )
            )

    def _set_exception(self, exception: Exception | None = None):
        if exception is None:
            self._exception = self._default_exception
            self._log_fun = lambda x: None
        else:
            self._exception = exception
            self._log_fun = PlastidFeature._log_fun

    def status_str(self) -> str:
        """
        Retrieves a string that describes the status of the contained `SeqRecord`.

        Returns:
            A string describing the `SeqRecord` status.

        """
        message = (
            f"skipped due to {self._exception}" if self._exception else "successful"
        )
        status = (
            f"parsing of {self._type} '{self.feat_name}' in {self.rec_name} {message}"
        )
        return status

    def log_exception(self):
        """
        Logs a string that describes the exception status of the contained `SeqRecord`.
        If the `SeqRecord` was successfully extracted, or if the issue that prevented
        extraction is not noteworthy, nothing will be logged. To unconditionally access
        a string representation of the `SeqRecord` for logging purposes, use `status_str()`.
        """
        if self._exception:
            self._log_fun(self.status_str())

    def get_record(self) -> Optional[SeqRecord]:
        """
        Instantiates the contained `SeqRecord` and returns it.
        If the `SeqRecord` could not be successfully extracted, `None` is returned.
        Additional information about the `None` return can be accessed by `status_str()`.

        Returns:
            The instantiated `SeqRecord`.

        """
        if self.seq_obj:
            return SeqRecord(self.seq_obj, id=self.seq_name, name="", description="")
        return None


class GeneFeature(PlastidFeature):
    """
    Primarily a wrapper for a gene `SeqRecord` that can be accessed by `get_record()`.
    It contains other variables and methods that document the creation of the `SeqRecord`.
    """

    _type = "gene"
    _default_exception = (
        "potential reading frame error of this feature in input file; "
        "associated protein will be skipped as well"
    )

    def _set_seq_obj(self, record: SeqRecord):
        super()._set_seq_obj(record)
        self._trim_mult_three()

    def _trim_mult_three(self):
        """Validate CDS frame / stops via :class:`CdsCodingValidator`.

        CDS-path only (``GeneFeature``). Non-triplet and premature-stop failures
        are Tier-1 quarantined; ATG trailing-codon trim is preserved; pseudogenes
        are excluded with ``INFO_PSEUDOGENE_EXCLUDED``.
        """
        if self.seq_obj is None:  # extraction failed; the exception is recorded
            return
        seq, reason = CdsCodingValidator.validate(self.seq_obj, self.feature)
        if reason is not None:
            loc = self._location_for_span()
            observed = len(self.seq_obj) if self.seq_obj is not None else 0
            self._reject_qc(reason, location_raw_coords(loc), observed)
            return
        self.seq_obj = seq


class ProteinFeature(GeneFeature):
    _type = "protein"
    _default_exception = "potential reading frame error of this feature in input file"

    def __init__(
        self,
        record: SeqRecord = None,
        feature: SeqFeature = None,
        gene: GeneFeature = None,
    ):
        """
        Primarily a wrapper for a protein `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        If an existing gene feature (`gene`) is provided instead of a `Biopython` `record` and `feature`,
        the object will be copied and the contained nucleotide sequence will be translated to a protein sequence.
        This is how the class is used within the module, but the standard `PlastidFeature` constructor
        exists for use cases where the nucleotide sequence of a feature is not used.

        Args:
            record: The record that the feature is being extracted from (default is `None`).
            feature: The feature to be extracted (default is `None`).
            gene: An existing gene feature (default is `None`).
        """
        # if we provide a GeneFeature to the constructor, we can just copy the attributes
        if gene is not None:
            self.__dict__.update(gene.__dict__)
        else:
            super().__init__(record, feature)
        self._set_prot_obj()

    def _set_prot_obj(self):
        if self.seq_obj is None:
            self._set_exception()
            return

        try:
            self.seq_obj = self.seq_obj.translate(
                table=11
            )  # Getting error TTA is not stop codon.
        except Exception as e:
            self._set_exception(e)


class IntronFeature(PlastidFeature):
    _type = "intron"

    def __init__(self, record: SeqRecord, feature: SeqFeature, offset: int = 0):
        """
        Primarily a wrapper for an intron `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        Args:
            record: The record that the feature is being extracted from.
            feature: The feature to be extracted.
            offset: An index offset for which intron is being considered (default is 0).
        """
        self.offset = offset
        super().__init__(record, feature)

    def _set_feat_name(self, feature: SeqFeature):
        super()._set_feat_name(feature)
        self.feat_name += "_intron" + str(self.offset + 1)

    def _set_feature(self, feature: SeqFeature):
        super()._set_feature(feature)
        exon_1 = self.feature.location.parts[self.offset]
        exon_2 = self.feature.location.parts[self.offset + 1]
        loc = circular_intron_bounds(exon_1, exon_2, self._parent_len, self._topology)
        if loc is None:
            # Long-arc / non-circular wrap → topology inversion (poison pill).
            try:
                long_arc = abs(int(exon_1.end) - int(exon_2.start))
            except Exception:
                long_arc = 0
            self._qc_reason = ERR_TOPOLOGY_INVERSION
            self._qc_coords = f"{exon_1}||{exon_2}"
            self._qc_observed_len = long_arc
            self.feature = None
            return
        self.feature.location = loc

    def _set_seq_obj(self, record: SeqRecord):
        if self.feature is None:
            if self._qc_reason is None:
                self._reject_qc(ERR_TOPOLOGY_INVERSION, "", 0)
            else:
                self.seq_obj = None
                self._set_exception(ValueError(self._qc_reason))
            return
        super()._set_seq_obj(record)


class IntergenicFeature(PlastidFeature):
    _type = "intergenic spacer"
    _default_exception = "negative intergenic length in input file"

    def __init__(
        self, record: SeqRecord, current_feat: SeqFeature, subsequent_feat: SeqFeature
    ):
        """
        Primarily a wrapper for an intergenic spacer `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        Args:
            record: The record that the feature is being extracted from.
            current_feat: The feature that borders the beginning of the intergenic space.
            subsequent_feat: The feature that borders the end of the intergenic space.
        """
        self._set_sub_feat(subsequent_feat)
        super().__init__(record, current_feat)

    def _set_sub_feat(self, subsequent_feat: SeqFeature):
        self.subsequent_feat = subsequent_feat
        self.subsequent_name = self.name_cleaner.get_safe_gene(subsequent_feat)

    def _set_feature(self, current_feat: SeqFeature):
        # Note: It's unclear if +1 is needed here.
        self.start_pos = ExactPosition(current_feat.location.end)  # +1)
        self.end_pos = ExactPosition(self.subsequent_feat.location.start)
        self.feature = (
            FeatureLocation(self.start_pos, self.end_pos)
            if self.start_pos < self.end_pos
            else None
        )
        self.default_exception = (
            f"{IntergenicFeature._default_exception}"
            f" (start pos: {self.start_pos}, end pos:{self.end_pos})"
        )

    def _set_seq_obj(self, record: SeqRecord):
        if self.feature is None:
            self._set_exception()
        else:
            super()._set_seq_obj(record)

    def _set_seq_name(self):
        self.inv_feat_name = f"{self.subsequent_name}_{self.feat_name}"
        self.feat_name += "_" + self.subsequent_name
        super()._set_seq_name()

VALID_PLASTID_STARTS = frozenset({"ATG", "GTG", "TTG", "CTG", "ATT", "ATC", "ATA", "ACG"})

def harmonize_5prime_drift(seq: str, locus: str, max_drift_codons: int = 40) -> str:
    """Trim upstream 5' annotation drift back to canonical plastid start codon."""
    if not seq or len(seq) < 9:
        return seq

    start = seq[:3].upper()
    if start in VALID_PLASTID_STARTS:
        return seq

    # Scan in-frame up to max_drift_codons
    limit = min(len(seq) - 3, max_drift_codons * 3)
    for offset in range(3, limit, 3):
        candidate = seq[offset:offset + 3].upper()
        if candidate in VALID_PLASTID_STARTS:
            subseq = seq[offset:]
            # Ensure remainder has no internal stops under Table 11
            from Bio.Seq import Seq
            prot = str(Seq(subseq).translate(table=11))
            if prot[:-1].count("*") == 0:
                return subseq
    return seq
