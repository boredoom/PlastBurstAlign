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
