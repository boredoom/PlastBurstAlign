from Bio.SeqFeature import FeatureLocation
from plastburstalign.qc import circular_intron_bounds, validate_cds_invariants


def test_negative_strand_linear_intron_geometry():
    """Reproduce NC_072203 atpF intron: strand=-1, parts in transcript order."""
    parent_len = 188863
    # Exon 1 (5' on - strand): 19667..19812
    exon1 = FeatureLocation(19667, 19812, strand=-1)
    # Exon 2 (3' on - strand): 18550..18960
    exon2 = FeatureLocation(18550, 18960, strand=-1)

    loc = circular_intron_bounds(exon1, exon2, parent_len, topology="circular")
    assert loc is not None
    # Must yield the 707 bp linear gap, NOT the 187,601 bp outer arc
    assert len(loc) == 707
    assert int(loc.start) == 18960
    assert int(loc.end) == 19667
    assert loc.strand == -1


def test_genuine_origin_wrap_intron():
    """Test synthetic gene straddling origin 0 on circular replicon."""
    parent_len = 150000
    exon1_gap = FeatureLocation(149500, 149800, strand=1)
    exon2_gap = FeatureLocation(200, 500, strand=1)
    loc_wrap = circular_intron_bounds(exon1_gap, exon2_gap, parent_len, topology="circular")
    assert loc_wrap is not None
    assert len(loc_wrap) == 400  # (150000 - 149800) + 200 = 400 bp


def test_cds_biological_invariants():
    """Verify in-frame clade expansions pass and frameshifts fail."""
    # Valid in-frame expansion (divisible by 3, ATG start, TAA stop, no internal stops)
    valid_cds = "ATG" + "GAA" * 50 + "TAA"
    is_valid, reason = validate_cds_invariants(valid_cds)
    assert is_valid is True
    assert reason is None

    # Frameshift (not triplet)
    frameshift_cds = "ATG" + "GAA" * 50 + "TA"
    is_valid, reason = validate_cds_invariants(frameshift_cds)
    assert is_valid is False
    assert reason == "ERR_FRAMESHIFT_NON_TRIPLET"

    # Premature internal stop codon
    stop_cds = "ATG" + "GAA" * 10 + "TAA" + "GAA" * 10 + "TAA"
    is_valid, reason = validate_cds_invariants(stop_cds)
    assert is_valid is False
    assert reason == "ERR_INTERNAL_STOP_CODON"


def test_clade_expansion_retention_in_cds():
    """Verify that real RefSeq expansions (e.g. rps18 492 bp, petD 555 bp) pass validation."""
    # Triraphis longipes rps18 sequence (NC_035532: 492 bp)
    rps18_poaceae = (
        "ATGTATATTAGCAAACAACCGTTTCGAAAATCGAAGCAACCGTTTCGAAAATCGAAGCAA"
        "ACATTTCATAAATCGAAGCAACCGTTTCGAAAATTCAAGCAACCGTTTCGAAAATCGAAG"
        "CAACCGTTTCGAAGACGATCGCGAATCGGCCCGGGAGATCGAATTGATTATCGTAATATG"
        "TCTTTAATTAATCGATTTATTTCGGAACAAGGAAAAATTTTAAGTCGCCGTATTAATCGA"
        "TTAACATTAAAACAACAGCGACTAATTACATTAGCGATTAAACAAGCGCGAATTTTAAGT"
        "TTTTTACCGTTTCGTAATTATGAAAATGAAAAACAATTTCAAGCGCAATCGATTTCAATT"
        "ATTACGGGATTACGACCGCGAAAAAATCGACATATTCCACAATTAACACAAAAATTTAAT"
        "AGTAATCGAAATTTAAGAAATTCGAATCAAAATTTAAGAAATAATAATCGTTCGTTATCA"
        "TCAGATTGTTAA"
    )
    is_valid, reason = validate_cds_invariants(rps18_poaceae)
    assert is_valid is True
    assert reason is None
    assert len(rps18_poaceae) == 492

def test_table11_isoleucine_and_rna_edited_starts():
    """Verify Table 11 alternate starts (ATA, ATC, ATT) and RNA-edited ACG pass."""
    for start in ["ATG", "GTG", "TTG", "CTG", "ATT", "ATC", "ATA", "ACG"]:
        # 60 bp in-frame coding sequence (> 45 bp minimal threshold)
        seq = start + ("GAA" * 18) + "TAA"
        valid, err = validate_cds_invariants(seq)
        assert valid is True, f"Failed on valid start codon {start}: {err}"

def test_gene_name_cleaner_product_fallback():
    """Verify that features lacking /gene fall back cleanly to /product."""
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    from plastburstalign.seqfeature_ops import GeneNameCleaner

    # Test 1: Feature with explicit gene qualifier
    feat_gene = SeqFeature(FeatureLocation(0, 100), type="CDS", qualifiers={"gene": ["rbcL"]})
    assert GeneNameCleaner.get_gene(feat_gene) == "rbcL"

    # Test 2: Feature with only /product (e.g., NC_012927 style)
    feat_prod = SeqFeature(
        FeatureLocation(0, 100),
        type="CDS",
        qualifiers={"product": ["ribosomal protein S12"]}
    )
    assert GeneNameCleaner.get_gene(feat_prod) == "rps12"

    # Test 3: Complex /product mappings
    mappings = [
        ("photosystem II protein D1", "psbA"),
        ("maturase K", "matK"),
        ("RNA polymerase beta'' subunit", "rpoC2"),
        ("ATP-dependent Clp protease proteolytic subunit", "clpP"),
        ("ATP synthase CF1 alpha subunit", "atpA"),
        ("cytochrome b6", "petB"),
        ("Psal", "psaI"),
    ]
    for prod_str, expected_gene in mappings:
        f = SeqFeature(FeatureLocation(0, 100), type="CDS", qualifiers={"product": [prod_str]})
        assert GeneNameCleaner.get_gene(f) == expected_gene, f"Failed on product: {prod_str}"


def test_watchdog_timeout_high_scale_floor():
    """Verify that huge-N/work alignments receive a resilient minimum watchdog deadline."""
    from plastburstalign.qc import alignment_timeout_seconds

    # For tiny work on 100 taxa
    small_t = alignment_timeout_seconds(work=1e5, tier=0, alphabet="prot", leased_threads=1)
    assert small_t >= 60.0

    # For ultra-large work (e.g. 13.5k taxa)
    large_t = alignment_timeout_seconds(work=6.96e9, tier=3, alphabet="prot", leased_threads=1)
    assert large_t >= 600.0, f"Expected timeout >= 600s, got {large_t}s"
