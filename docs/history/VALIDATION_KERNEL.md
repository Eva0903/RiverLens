# Kernel SHAP validation record

Environment: Python 3.12; Torch 2.5.1+cpu; SHAP 0.51.0; NumPy 2.3.5.

Eight tests passed without skips in the final run:

1. All 39 columns covered exactly once by 31 players; rain/wind encoding families kept joint.
2. Coalition replacements preserve entire trajectories and donor rows; empty/full endpoints use cached actual predictions; nonbinary masks rejected.
3. Kernel SHAP matches known affine-model attributions for multiple outputs.
4. Kernel SHAP matches exhaustive Shapley enumeration for a nonlinear interaction game.
5. An additive 31-player game retains all 31 nonzero attributions, checking that default top-ten feature selection is disabled.
6. Scope and fixed-budget protocol drift rejected.
7. End-to-end driver resume fixture using a saved two-layer LSTM: first invocation caches one origin; second invocation reuses it and completes the next origin; torch.autograd.grad is configured to raise if called; source checkpoint hashes remain unchanged.

The actual uploaded C_DO, C_NH3N, C_CODMN and C_TP seed42 checkpoints were also used with synthetic inputs at their original window/feature shapes. Each ran grouped Kernel SHAP for all six outputs with a full-rank 31-player design. Reconstruction residual was zero in all four checks. No backward calls were permitted. These checks do not constitute scientific attribution results or an estimator-convergence result.

Not executed here: original raw Excel reconstruction or CUDA on the user's machine; full formal 512-coalition/64-background/all-test analysis. Run audit and smoke locally before the formal job. CUDA inference uses exactly the existing precision-correct loader which previously passed the user's original cached-prediction check; no native-autograd forward adapter is invoked.

The former finite-difference failure is not bypassed within a gradient method: this package changes to a different, prediction-only explanation method. The new explanation game groups entire historical sequences and does not provide within-history temporal SHAP decomposition. All four models remain fixed at seed42.

8. A separate Python process blocks import of jinjiang.shap_supplement and still imports the Kernel SHAP entry module successfully. The isolated test folder contains no original 07 gradient modules.
