from typing import TypedDict


class CrossValSummary(TypedDict):
    run_count: int
    aggregation_mode: str
    
    final_model_mean_val_mba: float
    final_model_mean_test_mba: float
    final_model_mean_val_auroc: float
    final_model_mean_test_auroc: float
    final_model_std_val_mba: float
    final_model_std_test_mba: float
    final_model_std_val_auroc: float
    final_model_std_test_auroc: float

    best_model_mean_val_mba: float
    best_model_mean_test_mba: float
    best_model_mean_val_auroc: float
    best_model_mean_test_auroc: float
    best_model_std_val_mba: float
    best_model_std_test_mba: float
    best_model_std_val_auroc: float
    best_model_std_test_auroc: float
