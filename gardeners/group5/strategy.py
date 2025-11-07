# gardeners/group5/strategy.py
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from math import hypot

from core.garden import Garden
from core.micronutrients import Micronutrient
from core.plants.plant_variety import PlantVariety
from core.plants.species import Species
from core.point import Position


@dataclass
class _SimPlant:
    variety: PlantVariety
    x: float
    y: float
    inv: dict[Micronutrient, float]


class GreedyLocalStrategy:
    """
    One-by-one, nutrient-aware, dense placement strategy.

    Loop:
      1. figure out which nutrient is currently the weakest in the real garden,
      2. pick a species that produces that nutrient,
      3. try that species' varieties at every dense candidate position,
      4. for each hypothetical placement, run a tiny local nutrient sim,
      5. place the best-scoring one,
      6. repeat until nothing helps.
    """

    BASE_STEP = 0.5  # candidate grid spacing
    LOCAL_RADIUS = 7.0  # neighborhood radius for local sim
    LOCAL_SIM_STEPS = 4  # how many synthetic "days" to run per test

    def __init__(self, garden: Garden, varieties: list[PlantVariety]) -> None:
        self._garden = garden
        self._all_varieties = list(varieties)
        # bucket varieties by species once
        self._by_species: dict[Species, list[PlantVariety]] = {
            Species.RHODODENDRON: [],
            Species.GERANIUM: [],
            Species.BEGONIA: [],
        }
        for v in self._all_varieties:
            if v.species in self._by_species:
                self._by_species[v.species].append(v)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def cultivate(self) -> None:
        candidate_positions = self._build_dense_grid()

        # greedy loop
        while True:
            target_species = self._pick_species_to_help()
            species_vars = self._by_species.get(target_species, [])
            if not species_vars:
                break

            # order varieties of that species by local efficiency
            ordered_vars = sorted(
                species_vars,
                key=self._score_variety,
                reverse=True,
            )

            best_score = float('-inf')
            best_choice: tuple[Position, PlantVariety] | None = None

            # scan positions
            for pos in candidate_positions:
                # try the best variety first for this species
                placed_here = False
                for var in ordered_vars:
                    if not self._garden.can_place_plant(var, pos):
                        continue

                    score = self._simulate_local_and_score(pos, var)
                    if score > best_score:
                        best_score = score
                        best_choice = (pos, var)
                    placed_here = True
                    # we tried the top variety at this spot; move to next spot
                    break

                # if no variety of the chosen species fits here, just continue
                if not placed_here:
                    continue

            # nothing good found
            if best_choice is None or best_score <= 0:
                break

            pos, var = best_choice
            planted = self._garden.add_plant(var, pos)
            if planted is None:
                # unexpected failure, bail
                break

            # we can drop the exact position so we don't re-use it
            with suppress(ValueError):
                candidate_positions.remove(pos)

    # ------------------------------------------------------------------
    # candidate position utilities
    # ------------------------------------------------------------------
    def _build_dense_grid(self) -> list[Position]:
        pts: list[Position] = []
        step = self.BASE_STEP
        y = step
        while y < self._garden.height - step:
            x = step
            while x < self._garden.width - step:
                pts.append(Position(x, y))
                x += step
            y += step
        return pts

    # ------------------------------------------------------------------
    # choosing what to plant (nutrient-driven)
    # ------------------------------------------------------------------
    def _current_net_nutrients(self) -> dict[Micronutrient, float]:
        totals = {m: 0.0 for m in Micronutrient}
        for p in self._garden.plants:
            for m, amt in p.variety.nutrient_coefficients.items():
                totals[m] += amt
        return totals

    def _nutrient_to_species(self, nutrient: Micronutrient) -> Species:
        if nutrient == Micronutrient.R:
            return Species.RHODODENDRON
        if nutrient == Micronutrient.G:
            return Species.GERANIUM
        return Species.BEGONIA

    def _pick_species_to_help(self) -> Species:
        # no plants yet: start with red-producer
        if not self._garden.plants:
            return Species.RHODODENDRON

        nets = self._current_net_nutrients()
        weakest = min(nets, key=lambda m: nets[m])
        return self._nutrient_to_species(weakest)

    def _score_variety(self, v: PlantVariety) -> float:
        """
        Local intrinsic score: how good is this variety regardless of position?
        Prefer varieties that strongly produce their main nutrient and are small.
        """
        coeffs = v.nutrient_coefficients
        if v.species == Species.RHODODENDRON:
            prod = coeffs.get(Micronutrient.R, 0.0)
            cons = abs(coeffs.get(Micronutrient.G, 0.0)) + abs(coeffs.get(Micronutrient.B, 0.0))
        elif v.species == Species.GERANIUM:
            prod = coeffs.get(Micronutrient.G, 0.0)
            cons = abs(coeffs.get(Micronutrient.R, 0.0)) + abs(coeffs.get(Micronutrient.B, 0.0))
        else:
            prod = coeffs.get(Micronutrient.B, 0.0)
            cons = abs(coeffs.get(Micronutrient.R, 0.0)) + abs(coeffs.get(Micronutrient.G, 0.0))

        eff = prod / (cons + 0.001)
        size_bonus = 1.0 / (1.0 + v.radius)
        return eff * size_bonus

    # ------------------------------------------------------------------
    # local nutrient sim
    # ------------------------------------------------------------------
    def _dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return hypot(x1 - x2, y1 - y2)

    def _build_local_set(self, pos: Position, var: PlantVariety) -> list[_SimPlant]:
        sims: list[_SimPlant] = []
        # hypothetical new one (index 0)
        base_inv = {
            Micronutrient.R: 5.0 * var.radius,
            Micronutrient.G: 5.0 * var.radius,
            Micronutrient.B: 5.0 * var.radius,
        }
        sims.append(_SimPlant(variety=var, x=pos.x, y=pos.y, inv=base_inv))

        # nearby real plants
        for planted in self._garden.plants:
            d = self._dist(pos.x, pos.y, planted.position.x, planted.position.y)
            if d <= self.LOCAL_RADIUS:
                r = planted.variety.radius
                inv = {
                    Micronutrient.R: 5.0 * r,
                    Micronutrient.G: 5.0 * r,
                    Micronutrient.B: 5.0 * r,
                }
                sims.append(
                    _SimPlant(
                        variety=planted.variety,
                        x=planted.position.x,
                        y=planted.position.y,
                        inv=inv,
                    )
                )
        return sims

    def _produce_step(self, sims: list[_SimPlant]) -> None:
        for sp in sims:
            coeffs = sp.variety.nutrient_coefficients
            cap = 10.0 * sp.variety.radius
            # only apply if no component goes negative
            ok = True
            for m, delta in coeffs.items():
                if sp.inv[m] + delta < 0:
                    ok = False
                    break
            if not ok:
                continue
            for m, delta in coeffs.items():
                sp.inv[m] = min(cap, sp.inv[m] + delta)

    def _interaction_edges(self, sims: list[_SimPlant]) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        n = len(sims)
        for i in range(n):
            for j in range(i + 1, n):
                vi = sims[i].variety
                vj = sims[j].variety
                if vi.species == vj.species:
                    continue
                d = self._dist(sims[i].x, sims[i].y, sims[j].x, sims[j].y)
                if d < vi.radius + vj.radius:
                    edges.append((i, j))
        return edges

    def _exchange_step(self, sims: list[_SimPlant]) -> None:
        edges = self._interaction_edges(sims)

        producer: list[Micronutrient] = []
        neighbor_counts = [0] * len(sims)
        for sp in sims:
            if sp.variety.species == Species.RHODODENDRON:
                producer.append(Micronutrient.R)
            elif sp.variety.species == Species.GERANIUM:
                producer.append(Micronutrient.G)
            else:
                producer.append(Micronutrient.B)

        for i, j in edges:
            neighbor_counts[i] += 1
            neighbor_counts[j] += 1

        for i, j in edges:
            if neighbor_counts[i] == 0 or neighbor_counts[j] == 0:
                continue

            pi = producer[i]
            pj = producer[j]

            offer_i_total = sims[i].inv[pi] / 4.0
            offer_j_total = sims[j].inv[pj] / 4.0

            per_i = offer_i_total / neighbor_counts[i]
            per_j = offer_j_total / neighbor_counts[j]

            # "has more of what it gives than what it receives"
            if sims[i].inv[pi] <= sims[i].inv[pj]:
                continue
            if sims[j].inv[pj] <= sims[j].inv[pi]:
                continue

            q = min(per_i, per_j)
            if q <= 0:
                continue

            cap_i = 10.0 * sims[i].variety.radius
            cap_j = 10.0 * sims[j].variety.radius

            sims[i].inv[pi] -= q
            sims[i].inv[pj] = min(cap_i, sims[i].inv[pj] + q)

            sims[j].inv[pj] -= q
            sims[j].inv[pi] = min(cap_j, sims[j].inv[pi] + q)

    def _growth_step(self, sims: list[_SimPlant]) -> list[bool]:
        grew = [False] * len(sims)
        for idx, sp in enumerate(sims):
            r = sp.variety.radius
            need = 2.0 * r
            if (
                sp.inv[Micronutrient.R] >= need
                and sp.inv[Micronutrient.G] >= need
                and sp.inv[Micronutrient.B] >= need
            ):
                sp.inv[Micronutrient.R] -= r
                sp.inv[Micronutrient.G] -= r
                sp.inv[Micronutrient.B] -= r
                grew[idx] = True
        return grew

    def _simulate_local_and_score(self, pos: Position, var: PlantVariety) -> float:
        sims = self._build_local_set(pos, var)
        new_idx = 0

        total_neighbor_growth = 0
        new_grew = False

        for _ in range(self.LOCAL_SIM_STEPS):
            self._produce_step(sims)
            self._exchange_step(sims)
            grew_flags = self._growth_step(sims)
            if grew_flags[new_idx]:
                new_grew = True
            total_neighbor_growth += sum(grew_flags[1:])

        # measure balance of the local set after sim
        net_after = {Micronutrient.R: 0.0, Micronutrient.G: 0.0, Micronutrient.B: 0.0}
        for sp in sims:
            for m, delta in sp.variety.nutrient_coefficients.items():
                net_after[m] += delta
        balance_component = min(net_after.values())

        score = 0.0
        if new_grew:
            score += 100.0
        score += 5.0 * total_neighbor_growth
        score += balance_component

        return score
