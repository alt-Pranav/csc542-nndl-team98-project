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

### Prior Experience
Our team has prior experience training and evaluating CNNs for image classification in PyTorch. Applying mechanistic interpretability — specifically feature visualization and activation ablation — to a dense prediction network such as U-Net is novel territory for all team members.

### References

[1] Montgomery County X-ray Set, National Library of Medicine. https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MontgomeryCounty.md

[2] Jha, D. et al. "Kvasir-SEG: A Segmented Polyp Dataset." MMM 2020. https://arxiv.org/abs/1911.07069

[3] Antonelli, M. et al. "The Medical Segmentation Decathlon." Nature Communications, 2022. http://medicaldecathlon.com/

[4] Olah, C. et al. "Zoom In: An Introduction to Circuits." Distill, 2020. https://distill.pub/2020/circuits/zoom-in/
