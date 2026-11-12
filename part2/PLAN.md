# Fixed Part 2 experiment plan

Share the course448 protocol and released Part 1 evaluation. DINO: six categories x shots 1/2/4/8 x seeds 42/43/44 = 72 groups. CLIP: six zero-shot groups. References come only from normal fit rows and never validation/test. DINO uses official ViT-S/14, eight reference rotations, official PCA test masking, CPU FAISS exact search. CLIP uses ViT-L/14, MVTec-trained prompt weights, DPAM20/features24/sigma4. Freeze all model parameters; no VisA optimization.

Retain native and common image scores separately. Save complete validation/test indexes/maps, source identity, references, feature/build/search timings, memory-bank bytes and allocated GPU peaks. Verify 78 groups and 16,692 maps before publishing. Review a clean Git archive, then check remote branch/file bytes. Exclude datasets, official clones, environments, caches, full heatmaps and weights from Git. Never choose thresholds from test labels.
