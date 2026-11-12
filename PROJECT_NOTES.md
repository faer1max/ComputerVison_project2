# Course requirement reference and project scope

The authority is `AIAA3201_Project2_Few_Shot_Industrial_Visual_Anomaly_Detection_Spring2027.pdf` (six pages). Obtain the original from the course; it is not redistributed here. The technical checklist is [docs/PART1_PART2_REQUIREMENT_CHECKLIST.md](docs/PART1_PART2_REQUIREMENT_CHECKLIST.md).

All three parts are mandatory. Parts 1/2 provide classical normal-only baselines and official foundation-model reproduction. Part 3 must implement at least one justified improvement for a specific failure mode, retaining a frozen main backbone unless studying a small adapter. Negative improvement results still require controlled ablations and failure analysis.

Use VisA's required six categories, common splits/preprocessing/resolution/evaluation, normal-only threshold calibration, image AUROC/AP/FPR, pixel AUROC/AP/AUPRO, at least three random reference selections at 1/2/4/8 shots, resource measurements and at least 20 classified FP/FN cases. All twelve VisA categories are an extension, not the six-category minimum. Method variations are allowed when technically justified and the core task/protocol are satisfied.

Final deliverables remain separate: a 6-8 page CVPR LaTeX report excluding references, camera-ready arXiv upload with arXiv ID on the Canvas first page, public GitHub link at the abstract end, related work covering all ten recommended papers, methods/experiments/ablations/failures/conclusion, and a runnable or recorded demo. The specified demo package is `anomaly_demo.zip`. Do not submit raw code to Canvas. Do not omit Part 3 or claim that completing Parts 1/2 completes the whole course.
