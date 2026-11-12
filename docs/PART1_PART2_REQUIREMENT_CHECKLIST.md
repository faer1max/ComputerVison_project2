# Parts 1 and 2 requirement checklist

Authority: AIAA3201_Project2_Few_Shot_Industrial_Visual_Anomaly_Detection_Spring2027.pdf, pages 2-5. The original course PDF is retained locally, not redistributed. This checklist addresses Parts 1/2 only; Part 3, the final paper and final submission remain separate mandatory work.

| Requirement | Technical evidence and status |
| --- | --- |
| Normal template, ECC and difference map | Implemented; test-coordinate restoration, border fallback and retained registration failures |
| Normal-only convolutional AE, six classes | Implemented; six histories and 12 baseline metric rows |
| Normal-validation thresholds and strict ties | Passed; seven behavior tests and 648 unchanged-input decisions reproduced |
| Common-resolution brightness comparison | Completed; 324 normal images x 3 conditions x 2 methods = 1,944 predictions / 36 groups |
| Texture, structure and brightness interpretation | Completed in analysis document, with observed patterns separated from hypotheses |
| Official DINO/CLIP sources and weights | Pinned identities; official-entry sample parity checks retained |
| DINO 1/2/4/8-shot x three random seeds | Complete: seeds 42/43/44, six categories, 72 groups, 270 reference-use records |
| Means, sample SD and few-shot curves | Complete; within-seed category means then cross-seed ddof=1 |
| CLIP target-category zero-shot comparison | Six groups; MVTec-trained prompts, no VisA gradient training |
| Frozen-feature reasoning and coarse-patch limits | Analysis and two full-frame/detail-crop comparisons included |
| Shared split, geometry, resolution and evaluation | Protocol plus split/reference/RGB/normalization-inverse checks |
| Image AUROC/AP/FPR and pixel AUROC/AP/AUPRO | Both parts report them separately; AUPRO limit 0.30, 200 thresholds, 8-connectivity |
| Four-method comparison with native/common separation | 12-row table independently recomputed from fixed results |
| At least 20 categorized failures | 24 distinct Part 1 images, per-case observations and tentative mechanisms; Part 2 cases separately indexed |
| Extraction/scoring time, GPU peak and bank size | Records and stage boundaries retained; no unsupported isolated FPS ranking |
| Download/setup instructions for excluded files | PREPARATION covers pinned clones, official data/weights, metadata preparation and reconstruction of project outputs |
| Python, dependencies and execution commands | Python 3.11; matched CUDA torch/torchvision; isolated Windows Part 2 NumPy/FAISS |
| Single-image score, OK/NG and heatmap demo | Both baseline demos rerun successfully without input labels |
| Preserve historical numerical experiments | English translation checked structurally; historical hashes retained with explicit mapping |

## Verified supplementary results

Template pooled normal-validation FPR at brightness 0.75/1.0/1.25: 14.8148% / 4.3210% / 41.9753%. AE: 50.9259% / 4.3210% / 99.6914%. The normal 1.0 condition reproduces all 648 original decisions. This is a fixed-threshold synthetic brightness test, including clipping for brightened images, not a defect-recall or real-lighting generalization study.

## Review limits

The historical full Part 1 checks and Part 2 78-group/16,692-map checks are retained; an English release is not described as full retraining. Translation/provenance behavior and actual release checks are documented separately in SOURCE_PROVENANCE.md. No test label selects thresholds, references or epochs. Failure classifications are evidence-backed interpretations, not causal proof or an author signature. Existing official assets are checked; a completely new machine and a full redownload are not claimed as tested. Resource records are not an isolated end-to-end speed comparison.

All three course parts remain mandatory; this checklist does not claim completion of Part 3, the 6-8 page CVPR paper, arXiv/Canvas submission or final anomaly_demo.zip.
