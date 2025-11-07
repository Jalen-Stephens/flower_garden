# gardeners/group5/gardener.py
from __future__ import annotations

from core.garden import Garden
from core.gardener import Gardener
from core.plants.plant_variety import PlantVariety

from .strategy import GreedyLocalStrategy


class Gardener5(Gardener):
    def __init__(self, garden: Garden, varieties: list[PlantVariety]):
        super().__init__(garden, varieties)
        self._strategy = GreedyLocalStrategy(garden, varieties)

    def cultivate_garden(self) -> None:
        # the engine calls this
        self._strategy.cultivate()
