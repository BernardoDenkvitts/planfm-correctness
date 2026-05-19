### Experiment 2 - No raw plan size features
Evaluated whether removing the raw plan size features improves model performance across families.

For ad_xgb_wl_delta, performance worsened in all splits. The degradation was most relevant in extrapolation: MAE increased from 0.170 to 0.184, RMSE increased from 0.243 to 0.261, and R² dropped from 0.337 to 0.236.

For dd_lstm_shortest_path_delta, the results were almost the same as the baseline.

For dd_xgb_wl_delta results were almost the same as the baseline, R² validation and extrapolation decreased ~0.03, others also slightly decreased.

For ad_lstm_wl_delta, the modification significantly improved extrapolation generalization: R² increased from 0.178 to 0.310 and RMSE decreased from 0.270 to 0.248, despite a minor drop in validation performance.

### Experiment 3 - New ratio features
Evaluated whether adding new ratio features improves model performance across families. Overall, the results show that these features do not improve generalization and often degrade extrapolation performance, especially R².

For ad_lstm_wl_delta, validation and interpolation improved slightly, but extrapolation performance degraded severely

For ad_xgb_wl_delta, the new features produced a small improvement on interpolation, with R² increasing from 0.599 to 0.625. Extrapolation performance worsened considerably.

For dd_lstm_shortest_path_delta, validation and interpolation remained practically unchanged compared with the baseline. Extrapolation degraded strongly.

For dd_xgb_wl_delta, performance worsened across validation, interpolation, and extrapolation.

### Experiment 4 - SILU and GELU activation functions
Evaluated whether using SILU or GELU as activation functions instead of ReLU improves model generalization.

* For GELU:
    * For ad_lstm_wl_delta validation and interpolation metrics kept the same, test extrapolation metrics improved ~0.02 for MAE and RMSE, and ~0.12 for R².

    * For ad_xgb_wl_deta metrics kept the same for val, inter and extrapolation, minor changes ~0.01.

    * For dd_lstm_shortest_path_delta metrics kept the same for kept the same for val, inter and extrapolation, extrapolation R² increased ~0.02 
    
    * For dd_xgb_wl_deta metrics kept the same for val, inter and extrapolation.

* For SILU:
    * For ad_lstm_wl_delta validation and interpolation metrics kept the same, test extrapolation metrics improved ~0.02 for MAE and RMSE, and ~0.12 for R².

    * For ad_xgb_wl_deta metrics kept the same for val, inter and extrapolation, interpolation r² increased ~0.05.

    * For dd_lstm_shortest_path_delta metrics for val and interpolation kept the same, extrapolation all decreased, especially R² (0.36 to 0.32).
    
    * For dd_xgb_wl_deta kept the same slighly changes ~0.01.

### Experiment 5 - Removed High Correlated Features

For ad_lstm_wl_delta results did not change at all. Extrapolation R² decreased ~0.07. 

For ad_xgb_wl_deta metrics remained practically unchanged, extrapolation degraded substantially, with R² dropping from 0.337 to 0.167

For dd_lstm_shortest_path_delta R² validation decreased ~0.02, extrapolation decreased in all metrics, R² decreased ~0.13.

For dd_xgb_wl_delta results very close to the baseline, validation and interpolation metrics worsened minimally, extrapolation kept same.

### Experiment 6 only for ad_lstm_wl_delta - GELU + no raw plan size
Evaluated whether combining the two best interventions for this family (removing raw plan size features + using GELU) would improve results.
Combination resulted in underfitting and instability. Extrapolation R² collapsed back to 0.175 with an exploding standard deviation of ±0.282. This shows that both interventions were likely correcting the same underlying issue (shortcut learning). 

### Final Ideas
Use GELU for all families since ad_lstm and dd_lstm improved, and for the others kept neutral.

