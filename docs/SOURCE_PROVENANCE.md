# English translation and historical experiment provenance

The numerical experiments were produced before the English handoff. Their source SHA256 values are preserved in the original run/evaluation records. Do not replace those hashes and imply the experiments were run under translated source files.

The Chinese version is retained locally as `chinese_release_4ca7c66.zip` outside the release tree and in the historical Git commit 4ca7c66325b1a01c5b26d76fe355f896b93a7d6f. The current published tree is English-only. Historical Git history is not rewritten or deleted.

`assets/semantic_translation_check.json` records a structural comparison of every preexisting Python file immediately after translation: ignoring docstrings, all AST structure and non-Chinese constants were identical; only Chinese human-readable message literals differed. Comments do not enter the Python AST. This checks translation scope, not numerical accuracy by itself.

Subsequent explicit provenance-only edits add `scripts/source_provenance.py`, strict mapped-source comparisons in verification scripts, and recognition of mapped historical cache identities in the Part 2 CLI. They do not change model inference, data selection, scoring or evaluation mathematics. The old `source.json` is not overwritten when a recognized historical cache is resumed.

`assets/source_translation_bridge.json` maps exact recorded source bytes (including known Git LF variants) to exact English-release bytes. A mismatched key set, unknown historical hash, unrelated changed file or changed published hash is rejected. This is an explicit compatibility list, not a blanket instruction to skip source checks. New experiments still record their actual English source hashes normally.

Unchanged configurations and historical metric tables remain unchanged. Release checks include the behavioral tests, source bridge rejection cases, split/input checks, full stored-output verification and representative model smoke inference. This evidence does not claim new full training or a fresh-machine installation. Future algorithm changes must use new output directories and their own provenance; do not expand the bridge to conceal substantive changes.
