# Part 1 release checks

- Compile and run all seven behavioral tests.
- Verify six categories for both methods, normal-only fit/validation and unchanged official test rows.
- Require 448x448 finite float32 maps, retained failed registrations and test-coordinate maps.
- Check checkpoint epoch/seed/training history and normal-validation selection.
- Recompute image scores/metrics and validate pixel metric ranges, AUPRO settings and source hashes.
- Keep raw arrays/checkpoints local; publish metrics, metadata, histories and representative panels.
- Include at least 20 distinct failures with observed patterns and tentative explanations separated.
- Preserve the old 256 outputs separately; do not mix resolutions or silently retune on test labels.
- The English source bridge is an explicit audited translation mapping, not permission to ignore unrelated source changes.

The shared supplementary checklist covers brightness results and cross-method analysis.
