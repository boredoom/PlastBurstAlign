"""
Main execution coordinator for PlastBurstAlign with cluster execution support.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

from .alignment_ops import AlignmentCoordination
from .cluster_ops import (
    ClusterExecutionError,
    ClusterManifest,
    ClusterPartitionEngine,
    PartitionReceipt,
    SlurmScriptGenerator,
)
from .extraction_ops import DataCleaning, ExtractAndCollect
from .logging_ops import logger as log
from .seqfeature_ops import PlastidData
from .user_parameters import UserParameters


class PlastBurstAlign:
    def __init__(self, user_params: dict[str, Any] | None = None):
        """Coordinates extraction of sequence annotations from GenBank flatfiles,
        the alignment of those sequences grouped by region name,
        and writing the concatenated MSA results to file in NEXUS and FASTA formats.

        Args:
            user_params: Specifications for how the execution should be performed.
        """
        if not user_params or not isinstance(user_params, dict):
            self.user_params: dict[str, Any] = UserParameters()
        else:
            self.user_params = user_params

        # In cluster worker or finalize mode, bypass raw GenBank file inspection
        is_worker_or_final = (
            self.user_params.get("align_partition") is not None
            or self.user_params.get("finalize_slurm") is True
        )
        if not is_worker_or_final:
            self.plastid_data = PlastidData(self.user_params)
        else:
            self.plastid_data = None

    def execute(self):
        """Dispatches execution based on user parameters."""
        if self.user_params.get("align_partition") is not None:
            self._execute_worker()
        elif self.user_params.get("finalize_slurm"):
            self._execute_finalize()
        elif self.user_params.get("export_slurm"):
            self._execute_export()
        else:
            self._execute_monolithic()

    def _execute_monolithic(self):
        """Standard single-node monolithic execution."""
        extractor = ExtractAndCollect(self.plastid_data, self.user_params)
        extractor.extract()

        cleaner = DataCleaning(self.plastid_data, self.user_params)
        cleaner.clean()

        aligncoord = AlignmentCoordination(self.plastid_data, self.user_params)
        aligncoord.perform_MSAs()
        if self.user_params.get("concat"):
            aligncoord.concat_MSAs()

    def _execute_export(self):
        """Extracts and cleans features, exports partition manifest and Slurm job scripts."""
        extractor = ExtractAndCollect(self.plastid_data, self.user_params)
        extractor.extract()

        cleaner = DataCleaning(self.plastid_data, self.user_params)
        cleaner.clean()

        out_dir = self.user_params.get("out_dir")
        num_partitions = self.user_params.get("slurm_partitions", 10)
        mode = self.user_params.get("select_mode")
        prefix = "prot" if mode.collects_proteins() else "nucl"
        regions = list(self.plastid_data.nucleotides.keys())

        log.info(
            f"Exporting Slurm cluster manifest for {len(regions)} regions into "
            f"{num_partitions} partitions..."
        )

        partitions = ClusterPartitionEngine.partition_regions(
            regions=regions,
            out_dir=out_dir,
            num_partitions=num_partitions,
            prefix=prefix,
        )

        manifest_partitions: dict[str, dict[str, Any]] = {}
        for p in partitions:
            manifest_partitions[str(p.partition_id)] = {
                "regions": p.regions,
                "estimated_bytes": p.estimated_bytes,
            }

        manifest = ClusterManifest(
            version="1.0",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            select_mode=str(mode),
            out_dir=os.path.abspath(out_dir),
            total_regions=len(regions),
            num_partitions=len(partitions),
            alignment_tool=self.user_params.get("alignment_tool", "mafft"),
            alignment_tool_path=self.user_params.get("alignment_tool_path"),
            alignment_command=self.user_params.get("alignment_command"),
            adaptive_heuristics=self.user_params.get("adaptive_heuristics", True),
            partitions=manifest_partitions,
        )

        manifest_path = os.path.join(out_dir, "slurm_manifest.json")
        ClusterPartitionEngine.save_manifest(manifest, manifest_path)

        SlurmScriptGenerator.generate_scripts(
            out_dir=out_dir,
            manifest_path=manifest_path,
            num_partitions=len(partitions),
            cpus_per_task=self.user_params.get("slurm_cpus", 8),
            mem_per_task=self.user_params.get("slurm_mem", "16G"),
            time_limit=self.user_params.get("slurm_time", "04:00:00"),
            slurm_partition=self.user_params.get("slurm_partition"),
        )
        log.info("Slurm cluster export complete.")

    def _execute_worker(self):
        """Worker task execution: aligns assigned partition regions."""
        out_dir = self.user_params.get("out_dir")
        partition_id = self.user_params.get("align_partition")
        manifest_path = self.user_params.get("manifest_path") or os.path.join(
            out_dir, "slurm_manifest.json"
        )

        manifest = ClusterPartitionEngine.load_manifest(manifest_path)
        p_key = str(partition_id)
        if p_key not in manifest.partitions:
            raise ClusterExecutionError(
                f"Partition ID {partition_id} not found in manifest: {manifest_path}"
            )

        assigned_regions = manifest.partitions[p_key]["regions"]
        log.info(
            f"Worker running partition {partition_id} ({len(assigned_regions)} regions)..."
        )

        # Initialize minimal PlastidData connecting to the existing SQLite store
        self.plastid_data = PlastidData(self.user_params)
        aligncoord = AlignmentCoordination(self.plastid_data, self.user_params)

        successful_regions = aligncoord.align_partition_regions(assigned_regions)

        # Record atomic receipt
        PartitionReceipt.write_receipt(
            out_dir=out_dir,
            partition_id=partition_id,
            regions=assigned_regions,
            successful_regions=successful_regions,
        )
        log.info(
            f"Partition {partition_id} complete. "
            f"Successfully aligned {len(successful_regions)}/{len(assigned_regions)} regions."
        )

    def _execute_finalize(self):
        """Finalizer task execution: validates receipts and concatenates supermatrices."""
        out_dir = self.user_params.get("out_dir")
        manifest_path = self.user_params.get("manifest_path") or os.path.join(
            out_dir, "slurm_manifest.json"
        )

        manifest = ClusterPartitionEngine.load_manifest(manifest_path)
        all_done, missing_ids, successful_regions = (
            PartitionReceipt.verify_all_completed(
                out_dir=out_dir,
                num_partitions=manifest.num_partitions,
            )
        )

        if not all_done:
            raise ClusterExecutionError(
                f"Cannot finalize: {len(missing_ids)} partitions incomplete: {missing_ids}"
            )

        log.info(
            f"All {manifest.num_partitions} partitions verified. "
            f"Total successful region alignments: {len(successful_regions)}"
        )

        self.plastid_data = PlastidData(self.user_params)
        aligncoord = AlignmentCoordination(self.plastid_data, self.user_params)
        aligncoord.collect_partition_nexus(successful_regions)
        aligncoord.concat_MSAs()
        log.info("Cluster finalization and concatenation completed successfully.")
