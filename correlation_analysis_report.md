# Bottleneck Channel Correlation Analysis Report

## Summary

This report extends the Baby U-Net bottleneck interpretability analysis beyond single-channel visualization and ablation. The main finding is that several channels that looked visually similar in activation overlays are genuinely correlated in activation space. In particular, channels 16, 118, and 105 form a related negative-side bottleneck group: they activate over similar central thoracic / lower-lung regions and are moderately to strongly correlated across the validation split. Channel 60 behaves differently: ablating it improves validation IoU, and it is negatively correlated with the 16/118/105 group.

The key conclusion is that the bottleneck is not best interpreted as 128 independent features. Some functions are distributed across correlated channel groups, so cluster-level analysis is more informative than single-channel ranking alone.

## Files and Outputs Used

- Baseline metrics: `outputs/baby_unet_analysis_full/baseline_metrics.csv`
- Single-channel ablation: `outputs/baby_unet_analysis_full/bottleneck_ablation_val.csv`
- Ablation plot: `outputs/baby_unet_analysis_full/bottleneck_ablation_delta_iou.png`
- Harmful-channel overlays: `outputs/baby_unet_analysis_full/top_harmful_bottleneck_overlays.png`
- Strict correlation analysis: `outputs/channel_correlation/`
- Loose-threshold correlation analysis: `outputs/channel_correlation_t070/`

## Method

The analysis starts from the trained scratch Baby U-Net checkpoint `baby_unet_best.pth`. The model bottleneck has 128 channels at low spatial resolution, which makes it small enough for direct mechanistic analysis.

The workflow is:

1. Evaluate the trained model on validation and test splits using IoU and Dice.
2. Run single-channel ablation by zeroing one bottleneck channel at a time.
3. Collect bottleneck activations over the validation split.
4. Flatten each channel across all validation images and spatial locations.
5. Compute a channel-by-channel Pearson correlation matrix.
6. Cluster channels by thresholded activation correlation.
7. Compare clusters against single-channel ablation deltas.
8. Run cluster-level ablation for selected clusters to test whether distributed groups matter more than isolated channels.

This follows the spirit of Network Dissection: individual channels are treated as units whose activation maps can be inspected as evidence for semantic or anatomical concepts. It also uses the channel redundancy idea from pruning literature: highly correlated feature maps may represent overlapping functions, so a single-channel ablation can underestimate the importance of a concept. Finally, the cluster view is closer to concept-level interpretability methods such as TCAV, where the interpretable object is a direction or group in activation space rather than a single neuron.

## Task Performance

The trained Baby U-Net performs well on the Montgomery lung segmentation task:

- Validation IoU: 0.8665
- Validation Dice: 0.9271
- Test IoU: 0.8795
- Test Dice: 0.9353

The test performance is slightly higher than validation, suggesting that the held-out split is not harder than validation for this run. The model is accurate enough that bottleneck analysis is meaningful: the activations being interpreted correspond to a working segmentation model rather than a failed or underfit model.

## Single-Channel Ablation Results

Single-channel ablation showed that no individual bottleneck channel dominates the segmentation decision. The largest single-channel effects were small:

- Most harmful ablation: channel 118, delta IoU about -0.0010
- Other harmful channels: 25, 77, 16, 95, 84, 32, 117
- Most helpful/noisy channel to remove: channel 60, delta IoU about +0.0012

This is important for interpretation. A naive reading would say that no channel matters much. However, the activation overlays and sorted ablation curve suggest structure: harmful channels tend to share spatial motifs, especially central lower-lung or mediastinal/diaphragm-adjacent activation. This motivated correlation analysis.

## Correlation Results

The strict run used a correlation threshold of 0.85. It found 85 clusters, with the largest strict cluster containing 23 channels. At this threshold, channels 16, 118, and 105 are not merged into one cluster, but they are close neighbors:

- Corr(16, 105): 0.795
- Corr(16, 118): 0.720
- Corr(118, 105): 0.714

These values confirm that the visual similarity in the overlays is not just a one-image artifact. The channels share activation patterns across the validation split, but not strongly enough to pass a very strict 0.85 threshold.

The loose run used a correlation threshold of 0.70. At this threshold, channels 16, 118, and 105 fall into the same connected cluster. This loose cluster contains 42 channels:

`8 9 11 16 18 21 25 27 32 35 37 40 44 45 49 55 61 64 70 72 77 81 84 85 87 89 90 91 93 95 101 102 105 108 109 112 114 115 117 118 122 127`

This group is dominated by channels with negative or weak ablation deltas. Its mean single-channel delta IoU is approximately -0.000156, and it contains the strongest harmful channel, 118.

Channel 60 is clearly different:

- Corr(16, 60): -0.222
- Corr(118, 60): -0.281
- Corr(105, 60): -0.332

It is not part of the 16/118/105 group under either threshold. Its ablation improves validation IoU, suggesting that it may encode a competing or suppressive feature rather than a necessary lung-boundary feature.

## Cluster-Level Ablation

Cluster-level ablation gives the clearest evidence that the negative-side channels are functionally related. In the loose threshold run, ablating the cluster containing 16, 118, and 105 caused a much larger drop than any single channel:

- Cluster id: 1
- Cluster size: 42 channels
- Cluster ablation delta IoU: -0.2480
- Cluster ablation delta Dice: -0.1767

This should not be interpreted as all 42 channels representing exactly one clean concept. The low mean internal correlation of the large loose cluster shows that it is a connected component, not a compact all-to-all group. Still, the result is useful: channels 16, 118, and 105 sit inside a broader functional neighborhood whose removal significantly damages segmentation.

By contrast, ablating channel 60 alone improves IoU:

- Channel 60 delta IoU: about +0.0012
- Channel 60 delta Dice: about +0.0008

This supports treating channel 60 as a separate candidate for suppressive/noisy behavior, not as part of the harmful anatomical group.

## Interpretation

The negative-side channels appear to encode a central thoracic / diaphragm-adjacent activation pattern. This region is anatomically important because lung masks must separate the lung fields from the mediastinum, heart border, and diaphragm. The overlays for channels 16, 118, and 105 show similar activation concentration in this central lower region, and the correlation matrix confirms that they are related over the validation split.

The fact that individual deltas are small but group ablation is large suggests redundancy. The model can compensate when only one correlated channel is removed, but it fails when many related channels are removed together. This is a typical distributed-representation pattern: the interpretable feature is not a single channel, but a channel group.

The strict vs loose clustering comparison is also informative:

- Threshold 0.85 identifies compact, high-confidence redundant pairs or small groups.
- Threshold 0.70 identifies broader functional neighborhoods.

For the report and presentation, the strict threshold is better for conservative claims. The loose threshold is better for hypothesis generation and cluster ablation experiments.

## Discussion and Limitations

The analysis improves interpretability, but several caveats matter:

1. Correlation does not prove semantic identity. Channels 16, 118, and 105 are correlated and visually similar, but they may still respond to different sub-features of the same anatomical region.
2. Connected-component clustering at a loose threshold can produce large clusters where not every pair is highly correlated. The 42-channel cluster should be read as a functional neighborhood, not a single pure concept.
3. The validation split has only 20 images. Correlation estimates are useful but should be checked on the test split or with cross-validation.
4. Cluster ablation can overstate importance when many channels are removed at once, because it creates a larger distribution shift than single-channel ablation.
5. Current feature labels are descriptive rather than clinically verified. A radiology-informed review would be needed before naming channels as exact anatomical concepts.

## Recommended Next Steps

1. Repeat correlation clustering on the test split and compare cluster stability.
2. Add representative images per cluster, not only one sample, to avoid one-image interpretation.
3. Evaluate smaller subclusters around channels 16, 105, and 118, for example their top-5 nearest neighbors rather than the full loose 42-channel component.
4. Run signed cluster ablation separately for negative-delta and positive-delta channels inside the loose cluster.
5. Add mask-overlap scores between activation heatmaps and anatomical regions such as lung interior, lung boundary, central mediastinum, and diaphragm-adjacent pixels.

## Conclusion

The correlation analysis supports the hypothesis that Baby U-Net bottleneck interpretability should be performed at both channel and group levels. Channels 16, 118, and 105 are not isolated findings: they form a correlated neighborhood that likely encodes a shared central thoracic / lower-lung feature family. Channel 60 is a separate suppressive/noisy candidate whose removal slightly improves validation performance. Overall, the results make the interpretability story clearer: the model uses distributed bottleneck features, and cluster-level ablation exposes structure that single-channel ablation hides.
