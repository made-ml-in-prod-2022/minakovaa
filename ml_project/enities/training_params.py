from typing import Optional

from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str
    model_params: Optional[dict]
    is_use_ml_flow: bool = field(default=True)
