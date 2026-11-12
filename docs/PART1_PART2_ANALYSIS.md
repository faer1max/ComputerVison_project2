# Part 1 and Part 2: comparison and failure interpretation

## Shared inputs, different normal-data budgets

All four methods use the same six official test categories, normal fit/validation split, 448x448 RGB bilinear geometry, nearest/>0 masks and shared evaluator. The PDF requires consistency; 448 is this project's choice, not a uniquely mandated course size. Pretrained model normalizations remain explicit interface requirements.

The template uses up to the first 24 normal fit images, averaging successfully registered references. The AE uses all normal fit images for ten epochs and selects the lowest normal-validation MSE checkpoint. DINO uses 1/2/4/8 random normal-fit references, three seeds and eight rotations per reference. CLIP uses MVTec-trained prompts and no VisA gradient optimization. These budgets differ: this is not an equal-training-data ablation.

The unified table is `results/part12_comparison/macro_metrics.csv`. Common scores are the mean of the highest 1% of 448-map pixels. Native DINO patch scores and CLIP global image/text probabilities are separate rows. Each seed first averages the six categories; DINO SD is the sample SD across three seeds. Deterministic methods are not repeated merely to fabricate SD.

## Texture, structure and illumination

Pixel-template differences require correspondence and representative normal appearance. Euclidean ECC reduces global rotation/translation, but cannot eliminate changes in relative capsule arrangement, food shape, background texture or reflectance. C01/C05/C07/C09/C11 show normal outlines producing false alarms. This is evidence of appearance mismatch, not automatic proof of an ECC implementation defect.

The AE's normal reconstruction objective does not guarantee semantic anomaly recognition. C13/C15/C21/C23 show residuals along ordinary edges/reflections; C17/C18 show high-frequency background residuals. Structural damage in C19 and a thin defect in C22 fail to stand out from normal residuals. Learning normal appearance may help some patterns, but this architecture/training setting does not ensure residuals concentrate on defects.

Part 1's case table distinguishes small spots/thin lines/weak local appearance response (C02/C04/C06/C12/C14/C16/C18/C22/C24) from larger damage or missing structure (C08/C19/C20). These are observed groupings, not official defect-type labels; no unsupported stratified dataset-wide AUROC is claimed.

The new brightness experiment reuses saved 448 templates, best AE checkpoints and original normal thresholds. Exactly the same 324 normal validation images receive multipliers 0.75/1.0/1.25 followed by clipping and uint8 truncation. No threshold is recalibrated. The 1.0 condition reproduces all 648 original method/image decisions and scores. Original float32 CSV score representation is retained to avoid numerical flips at threshold ties; raw recomputed scores are also saved.

Pooled validation FPR for template at 0.75/1.0/1.25 is 0.148148/0.043210/0.419753. AE is 0.509259/0.043210/0.996914. The 1.25 condition includes highlight clipping. This experiment measures normal-image false alarms under synthetic brightness changes, not defect recall, real-camera illumination, viewpoint or domain-shift robustness. Six-category macro FPR and pooled FPR are both saved and are not interchangeable. Old 256-resolution brightness results are not mixed with this study.

## Why frozen features can localize unseen defects

DINOv2 maps local image regions into a pretrained feature space; reference patches describe normal target-product appearance. A test patch far from the normal bank can receive a large distance without defect labels or target-category gradient training. The model does not thereby know every VisA defect: uncovered normal variations can also be distant. Official PCA foreground selection and reference rotations are recorded method-specific steps.

CLIP compares image features with normal/abnormal text features in a shared space. The global image score and local patch map answer different questions; a local defect response may coexist with an image-level false negative. Object-agnostic prompts do not guarantee sensitivity to tiny material defects. Zero-shot model transfer and target normal-validation operating-point calibration are explicitly distinct.

## Coarse-patch limits and visual evidence

With patch width 14, a 448 input produces a 32x32 grid. A tiny spot or line can be mixed with surrounding appearance within a patch. Interpolation and Gaussian smoothing produce a dense map but do not create additional independent fine-detail evidence. A broad hotspot is not proof of exact small-defect boundaries; one failure does not establish patch width as its sole cause.

`capsules_detail.png` and `chewinggum_detail.png` show the fixed first anomalous image of each named category, official mask, DINO seed42/1-shot map and CLIP map at full-frame and GT-bounding-box crop scales. Crops are visualization only: they are not fed back to the model or used to recompute metrics. Crop color limits follow the whole image. Tiny capsule defects and larger chewinggum damage illustrate response extent and detail limitations, motivating future multi-scale experiments without claiming an unperformed ablation.

## Classified failures

All four Part 1 six-case panels were reviewed again on 2026-11-12: 24 distinct product images with input, GT, heatmap and thresholded map. C01-C12 expose template outline/arrangement/background mismatch and weak local defect response. C13-C24 expose AE normal-edge/background residuals and weak appearance/structural responses. The per-case table and numerical measurements separate observations from tentative mechanisms; assistant review does not impersonate author sign-off or causal validation.

Image FP/FN, pixel recall and background FPR must be interpreted separately. An image FN may still have a correctly located weak hotspot. High AUROC does not guarantee low misses at the fixed normal-FPR operating point. Part 2 has an additional 24 model-image examples; these are not claimed to be 24 unique product images. Test-case interpretation did not change thresholds, reference selection or training epochs.

## Efficiency and provenance

Published resource tables record feature/scoring times, allocated GPU peaks and memory-bank size. Neural feature extraction is inapplicable to the template method; normal reference-bank size is inapplicable to CLIP. DINO cache hits, interrupted cumulative extraction timers and partially concurrent stages limit timing interpretation. No isolated end-to-end FPS ranking is claimed.

Upstream code and weight identities are recorded in `assets/source_revisions.json` and `part2/assets_verified.json`. Images/masks originate from VisA/SPot-the-Difference; backbone features from DINOv2/CLIP and methods from their official repositories. Source links are in PREPARATION and literature/sources.json. The English source release retains historical experiment hashes through the explicit audited mapping; see SOURCE_PROVENANCE.md.

## Reproduce the supplementary study

After complete Part 1 and Part 2 outputs exist, set the documented asset/data/output variables if reusing existing directories, then run:

```powershell
python scripts/illumination_course448.py
python scripts/build_part12_comparison.py
python scripts/verify_part12_supplement.py
python part1/verify.py
python part2/check_contract.py
```

The scripts overwrite their own supplementary output names on an intentional rerun. Preserve historical outputs before changing actual algorithms or configurations.
