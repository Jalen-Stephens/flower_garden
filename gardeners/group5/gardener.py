from __future__ import annotations

from core.garden import Garden
from core.gardener import Gardener
from core.plants.plant_variety import PlantVariety

from .hybrid_strategy import HybridStrategy, HybridStrategyFlex, HybridStrategyBalanced
from .principled_strategy import PrincipledStrategy


class Gardener5(Gardener):
    def __init__(self, garden: Garden, varieties: list[PlantVariety]):
        super().__init__(garden, varieties)
        # Default now to PrincipledStrategy (cluster + deficit lattice + greedy balance)
        self._strategy = PrincipledStrategy(garden, varieties)

    def cultivate_garden(self) -> None:
        self._strategy.cultivate()