"""Domain operations and incremental data structures for NEXUS alignment data."""

from __future__ import annotations

import copy

from Bio.Nexus.Nexus import Nexus as NexusAlignment
from Bio.Seq import Seq


class ConcatenationError(RuntimeError):
    """Raised when alignment concatenation fails."""


class IncrementalNexusAccumulator:
    """Incrementally accumulates individual Nexus alignment matrices into a single

    concatenated Nexus alignment. Produces output byte-equivalent to Nexus.Nexus.combine()
    without holding all individual alignments in memory simultaneously.
    """

    def __init__(self) -> None:
        self.combined: NexusAlignment | None = None

    def add(self, name: str, m: NexusAlignment) -> None:
        if self.combined is None:
            self.combined = copy.deepcopy(m)
            self.combined.charlabels = None
            self.combined.statelabels = None
            self.combined.interleave = False
            self.combined.translate = None

            for cn, cs in list(self.combined.charsets.items()):
                self.combined.charsets[f"{name}.{cn}"] = cs
                del self.combined.charsets[cn]
            for tn, ts in list(self.combined.taxsets.items()):
                self.combined.taxsets[f"{name}.{tn}"] = ts
                del self.combined.taxsets[tn]

            self.combined.charpartitions = {"combined": {name: list(range(self.combined.nchar))}}
            return

        comb = self.combined
        assert comb is not None

        # Handle mixed datatypes across partitions
        if comb.datatype != m.datatype:
            comb.datatype = "None"

        comb_taxlabels: list[str] = comb.taxlabels
        m_taxlabels: list[str] = m.taxlabels

        both = [t for t in comb_taxlabels if t in m_taxlabels]
        combined_only = [t for t in comb_taxlabels if t not in both]
        m_only = [t for t in m_taxlabels if t not in both]

        assert isinstance(comb.matrix, dict)
        assert isinstance(m.matrix, dict)
        comb_matrix = comb.matrix
        m_matrix = m.matrix

        for t in both:
            comb_matrix[t] += Seq(
                str(m_matrix[t]).replace(m.gap, comb.gap).replace(m.missing, comb.missing)
            )
        for t in combined_only:
            comb_matrix[t] += Seq(comb.missing * m.nchar)
        for t in m_only:
            comb_matrix[t] = Seq(comb.missing * comb.nchar) + Seq(
                str(m_matrix[t]).replace(m.gap, comb.gap).replace(m.missing, comb.missing)
            )

        comb_taxlabels.extend(m_only)
        for cn, cs in m.charsets.items():
            comb.charsets[f"{name}.{cn}"] = [x + comb.nchar for x in cs]
        if m.taxsets:
            if not comb.taxsets:
                comb.taxsets = {}
            comb.taxsets.update({f"{name}.{tn}": ts for tn, ts in m.taxsets.items()})

        comb.charpartitions["combined"][name] = list(range(comb.nchar, comb.nchar + m.nchar))
        if m.charlabels:
            if not comb.charlabels:
                comb.charlabels = {}
            comb.charlabels.update({comb.nchar + i: label for i, label in m.charlabels.items()})
        comb.nchar += m.nchar
        comb.ntax += len(m_only)

    def finalize(self) -> NexusAlignment | None:
        if self.combined is None:
            return None
        charpartitions: dict[str, dict[str, list[int]]] = self.combined.charpartitions
        charsets: dict[str, list[int]] = self.combined.charsets
        for c, partition in charpartitions.get("combined", {}).items():
            charsets[c] = partition
        return self.combined
