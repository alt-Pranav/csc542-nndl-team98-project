# Unpacking the U-Net Bottleneck: Mechanistic Interpretability in Medical Image Segmentation

## By Team 98: Pranav Bhagwat, Raghav Kesari, and Tiehang Zhang

### Introduction
Deep learning has achieved remarkable performance in medical image analysis, yet these models largely operate as "black boxes." 
In clinical settings, opacity is a serious liability: models can learn spurious correlations — such as surgical markings or scanner watermarks — rather than genuine anatomical features. This project applies mechanistic interpretability to a segmentation model to reverse-engineer its internal representations. By understanding how raw pixels are translated into clinical features, we can verify whether the model is making decisions for the right reasons, increasing transparency and trust in automated diagnostics.

### Dataset
We will use the Montgomery County Chest X-ray Set [1], a publicly available dataset of 2D grayscale chest X-rays with ground-truth binary lung masks. This dataset is well-suited to our interpretability goals: the task is binary (lung vs. background), images are structurally consistent, and the domain is clinically meaningful. Alternative datasets considered include Kvasir-SEG [2], which offers 1,000 annotated colonoscopy images for polyp segmentation, and the Medical Segmentation Decathlon [3], a multi-task benchmark spanning 10 organ and lesion types across CT and MRI. Should the Montgomery dataset prove too small for robust training, Kvasir-SEG represents a natural drop-in alternative with comparable binary segmentation structure. The output in all cases is a 2D binary pixel-wise prediction mask (resized to 128×128 or 256×256), where each pixel is classified as either target structure or background.

### Model Architecture
We propose a custom CNN — specifically a tightly constrained "Baby U-Net" trained entirely from scratch. Pre-trained weights are deliberately avoided, as they introduce millions of entangled features (polysemanticity) that render mechanistic analysis intractable. The architecture will be limited to 3–4 downsampling blocks with a narrow bottleneck (at most 128 channels), keeping feature visualization and circuit tracing computationally feasible.

### Evaluation Plan
Evaluation proceeds in two phases. 

- **Phase 1 (Task Performance):** The trained model is evaluated on a held-out test set using Intersection over Union (IoU) and the Dice Coefficient, the standard metrics for spatial overlap in segmentation. 
- **Phase 2 (Interpretability):** We isolate the bottleneck layer and apply feature visualization [4] to map individual channels to human-interpretable concepts (e.g., "rib edge detector," "diaphragm curve detector"). Findings are validated quantitatively through targeted ablation: if a specific channel is identified as responsible for a class of false positives, zeroing it out must predictably correct the output mask.

### Correlation-Based Bottleneck Analysis
Single-channel ablation is useful but incomplete because U-Net bottleneck channels can be redundant: multiple channels may encode nearly the same spatial region, so ablating one channel at a time can underestimate the importance of the shared concept. To address this, we add a correlation-based evaluation step inspired by Network Dissection [5], channel redundancy analysis [6], and concept-level interpretability methods such as TCAV [7]. We collect bottleneck activations across the validation split, flatten each channel's activation maps across images and spatial positions, compute a channel-by-channel correlation matrix, and cluster channels by activation similarity. We then compare these clusters with the existing IoU/Dice ablation deltas and run group ablations for candidate clusters.

This analysis showed that several visually similar harmful channels, including channels 16, 118, and 105, are moderately to strongly correlated in activation space (pairwise correlations approximately 0.71-0.80), while channel 60 is negatively correlated with that group and improves validation IoU when ablated. At a stricter clustering threshold (0.85), channels 16, 118, and 105 remain separate but close neighbors; at a looser threshold (0.70), they join the same connected cluster. Group ablation of this loose negative-side cluster causes a much larger IoU drop than any single-channel ablation, indicating that the model distributes some anatomical bottleneck information across correlated channels rather than storing it in one isolated unit. This supports the project hypothesis that interpretability should be evaluated at both the individual-channel and channel-group level.

### Prior Experience
Our team has prior experience training and evaluating CNNs for image classification in PyTorch. Applying mechanistic interpretability — specifically feature visualization and activation ablation — to a dense prediction network such as U-Net is novel territory for all team members.

### References

[1] Montgomery County X-ray Set, National Library of Medicine. https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MontgomeryCounty.md

[2] Jha, D. et al. "Kvasir-SEG: A Segmented Polyp Dataset." MMM 2020. https://arxiv.org/abs/1911.07069

[3] Antonelli, M. et al. "The Medical Segmentation Decathlon." Nature Communications, 2022. http://medicaldecathlon.com/

[4] Olah, C. et al. "Zoom In: An Introduction to Circuits." Distill, 2020. https://distill.pub/2020/circuits/zoom-in/

[5] Bau, D. et al. "Network Dissection: Quantifying Interpretability of Deep Visual Representations." CVPR, 2017. https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html

[6] Suau, X. et al. "Principal Filter Analysis for Guided Network Compression." Journal of Mathematical Imaging and Vision, 2020. https://arxiv.org/abs/1807.10585

[7] Kim, B. et al. "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)." ICML, 2018. https://proceedings.mlr.press/v80/kim18d.html
