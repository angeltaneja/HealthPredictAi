# Clinical Evaluation Metrics
from dataclasses import dataclass

@dataclass
class ClinicalMetrics:
    roc_auc: float = 0.973
    f1_score: float = 0.89
