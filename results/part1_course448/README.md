# Verified Part 1 course448 summaries

`metrics.csv` contains 12 category/method rows. `macro_mean.csv` averages the six categories. Six training-history files document AE fitting. `run_metadata.json`, `evaluation_protocol.json`, `release_verification.json` and environment snapshots retain the historical execution identity and evaluation settings.

`visual_cases.csv` indexes 24 distinct images; `visual_case_statistics.csv` quantifies score margins, defect fraction, pixel recall and background FPR. Four `visual_cases_*.png` panels correspond to cases C01-C24. `FAILURE_ANALYSIS.md` separates observations from possible mechanisms. Error inventories are fixed-threshold outcomes, not threshold-selection data.

Demo JSON/PNG files are representative historical outputs, including genuine failures. Checkpoints, templates, raw validation/test CSVs and heatmaps are not included; regenerate them using Part 1 commands. The `illumination/` subdirectory contains the later normal-validation brightness study with fixed original thresholds. The English source mapping preserves historical hashes rather than pretending old results were newly trained.
