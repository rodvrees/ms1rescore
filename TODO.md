- Move MSF parser to psm_utils
- LC-MS/MS prior features should just be MS2Rescore features
- cleanup: classes of scoring algos? with a base class?
- New release of psm_utils
- Implement JSON config structure like in MS2Rescore
- Priority:
    1. Check features that are deemed to be important:
        - CHCA stuff
        - Monoisotope confidence
        - Isotope envelope (altough it looks okay)
    2. Check some stuff that looks off:
        - RT residual
        - log_mean_intensity and spatial_entropy

IM2Deep finetuning - is it using best val_loss as early stopping?