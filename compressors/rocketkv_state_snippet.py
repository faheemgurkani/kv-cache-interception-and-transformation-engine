@dataclass
class RocketKVLayerState:
    stage1_locked: bool = False
    permanent_prefix_global: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    current_global: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
