# Visual review of 24 distinct Part 1 failures

Images and official masks originate from VisA/SPot-the-Difference (Zou et al., 2022), CC BY 4.0. `visual_cases.csv` indexes distinct relative image paths, fixed scores and thresholds. `visual_case_statistics.csv` quantifies area, recall, background false alarms and score margins. Four panels cover C01-C06, C07-C12, C13-C18 and C19-C24 with input/GT/heatmap/thresholded prediction.

Reviewed again on 2026-11-12. Observations and possible mechanisms are separated; assistant review is not an author signature or a controlled causal experiment. FP/FN refers to the image decision, not necessarily pixel-localization success. Per-image color scales do not make colors comparable across images.

| Case | Method/error | Observation | Classification / possible mechanism |
| --- | --- | --- | --- |
| C01 | Template/candle FP | Normal candle arcs dominate the response. | Normal contour mismatch; mean-template position/shape differences may dominate RGB error. |
| C02 | Template/candle FN | The tiny annotated spot is weak; ordinary arcs dominate. | Small appearance defect; score aggregation may dilute local evidence. |
| C03 | Template/capsules FP | Normal highlights and capsule outlines produce many responses. | Normal arrangement/reflection differences not removed by global alignment. |
| C04 | Template/capsules FN | A tiny single-capsule defect is not prominent. | Fine defect response competes with ordinary highlights and arrangement differences. |
| C05 | Template/cashew FP | Normal object edges produce a strong outline. | Shape/contour variability relative to the mean template. |
| C06 | Template/cashew FN | Spot and thin-line annotations compete with background and contour response. | Weak local appearance contrast and background mismatch. |
| C07 | Template/chewinggum FP | Normal rectangular edges form a bright frame. | Normal position/shape mismatch and background texture. |
| C08 | Template/chewinggum FN | Damage is locally visible, while image score stays below threshold. | Local evidence and image-level aggregation/calibration disagree. |
| C09 | Template/fryum FP | Normal spokes and outer ring respond strongly. | Normal structural alignment/shape variability. |
| C10 | Template/fryum FN | Annotated local damage is not stronger than normal wheel boundaries. | Weak damage discrimination amid complex geometry. |
| C11 | Template/pipe_fryum FP | Normal rectangular boundary forms a bright frame. | Boundary/pose mismatch after global alignment. |
| C12 | Template/pipe_fryum FN | A small interior mark is weak while the outer frame dominates. | Small appearance defect masked by contour response. |
| C13 | AE/candle FP | Normal circumference, wicks and small residuals cross the pixel threshold. | Normal high-frequency reconstruction error. |
| C14 | AE/candle FN | Lower-right damage is weak; ordinary circumference remains visible. | Small damage response insufficiently separated from edge residuals. |
| C15 | AE/capsules FP | Normal reflections and capsule edges produce residuals. | Complex normal reflective appearance is not reconstructed precisely. |
| C16 | AE/capsules FN | Tiny single-capsule GT does not stand out from ordinary residuals. | Fine defect/reflection ambiguity; causal mechanism still needs an ablation. |
| C17 | AE/cashew FP | Many black-background texture residuals are predicted. | High-frequency background reconstruction residuals may inflate the image score. |
| C18 | AE/cashew FN | Two annotated spots lack concentrated response; background residuals are widespread. | Weak local appearance discrimination with background interference. |
| C19 | AE/chewinggum FN | A large upper-edge structural defect lacks a correspondingly strong response. | Reconstruction error does not guarantee normal-structure understanding. |
| C20 | AE/chewinggum FN | Lower damage and fine-line GT are only partially visible in the response. | Local evidence may not exceed the fixed image operating point. |
| C21 | AE/fryum FP | Normal ring and spoke boundaries are predicted. | Normal geometric edge reconstruction error. |
| C22 | AE/fryum FN | A thin annotated defect does not stand out from normal spokes. | Thin-line response and aggregation sensitivity. |
| C23 | AE/pipe_fryum FP | Normal rectangular edges form a predicted frame. | Boundary reconstruction residual and operating-point sensitivity. |
| C24 | AE/pipe_fryum FN | Small GT and interior residuals coexist while boundaries dominate. | Weak spot discrimination amid edge/texture residuals. |

These patterns motivate future controlled work on background suppression, fine detail and aggregation. They do not establish a proven improvement. No reference, threshold, training epoch or postprocessing parameter was retuned using these test cases.
