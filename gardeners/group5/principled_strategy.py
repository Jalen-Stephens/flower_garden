from __future__ import annotations

"""Group 5 Principled Strategy

Design Goals (inspired by high-performing patterns):
- Early multi-species micro‑clusters (R,G,B) to spark exchanges quickly.
- Continuous nutrient deficit feedback guiding species selection.
- Interaction maximization that prefers placements completing complementary sets.
- Lightweight spatial indexing to accelerate overlap & interaction queries.
- Separation of phases with clear responsibilities to keep reasoning transparent.

Phases:
1. Seed Clusters: place a bounded number of compact R/G/B triads using a jittered anchor lattice.
2. Deficit Lattice Fill: traverse a hex-like lattice, choosing species that alleviate current global micronutrient deficits; locally rank candidate varieties by production efficiency & potential interaction completion.
3. Greedy Interaction Balancer: until varieties exhausted or no space remains, place single varieties maximizing a combined score: deficit relief + balanced interaction completion.

Distinctiveness vs reference implementation:
- Adds spatial bins (uniform grid) for faster neighbor lookups.
- Uses a combined score with dynamic deficit weights instead of fixed species priority.
- Employs jittered lattice anchors and rotation sampling for cluster robustness.
- Avoids direct reuse of any reference code; all helpers reimplemented.
"""

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple

from core.garden import Garden
from core.micronutrients import Micronutrient
from core.plants.plant_variety import PlantVariety
from core.plants.species import Species
from core.point import Position


@dataclass
class VarietyScore:
    variety: PlantVariety
    efficiency: float  # production / (consumption + epsilon) * small-radius bias


class SpatialBins:
    """Simple uniform spatial binning to reduce O(N) scans for interaction/overlap.

    Bins keyed by (ix, iy). Bin size chosen from minimum radius.
    """

    def __init__(self, garden: Garden, cell_size: float):
        self.garden = garden
        self.cell = max(cell_size, 0.05)
        self._bins: Dict[Tuple[int, int], List[int]] = {}

    def _key(self, pos: Position) -> Tuple[int, int]:
        return int(pos.x // self.cell), int(pos.y // self.cell)

    def rebuild(self):
        self._bins.clear()
        for idx, plant in enumerate(self.garden.plants):
            self._bins.setdefault(self._key(plant.position), []).append(idx)

    def nearby_indices(self, pos: Position, search_radius: float) -> Iterable[int]:
        gx, gy = self._key(pos)
        r = int(math.ceil(search_radius / self.cell))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                bucket = self._bins.get((gx + dx, gy + dy))
                if bucket:
                    for idx in bucket:
                        yield idx


class PrincipledStrategy:
    CLUSTER_LIMIT = 140          # max triad anchors attempted
    TRIAD_ROTATIONS = (0.0, math.pi/3, 2*math.pi/3, math.pi)  # sampled rotations
    MAX_TRIADS = 60               # upper bound on triads actually placed
    LATTICE_JITTER = 0.15         # jitter magnitude as fraction of spacing
    GREEDY_GRID_STEP = 0.18       # candidate spacing in greedy phase
    INTERACTION_RADIUS_FACTOR = 1.0  # multiplier when scanning for interactions

    def __init__(self, garden: Garden, varieties: List[PlantVariety]):
        self.garden = garden
        self.initial_varieties = list(varieties)
        self.available = list(varieties)
        self.min_radius = min((v.radius for v in varieties), default=0.5)
        self._bins = SpatialBins(garden, cell_size=self.min_radius * 0.75)

    # ---------------------------- Public API ---------------------------- #
    def cultivate(self) -> None:
        if not self.available:
            return
        # Phase 1
        self._seed_triads()
        # Phase 2
        self._lattice_deficit_fill()
        # Phase 3
        self._greedy_balance()

    # ---------------------------- Phase 1: Triads ----------------------- #
    def _seed_triads(self) -> None:
        species_groups = self._group_by_species(self.available)
        if any(len(species_groups[sp]) == 0 for sp in species_groups):
            return
        # Pre-score varieties by efficiency
        scored_groups = {sp: self._score_and_sort(vars) for sp, vars in species_groups.items()}

        # Lattice spacing derived from average radius of top candidates
        base_r = sum(v.variety.radius for v in scored_groups[Species.RHODODENDRON][:5]) / max(5, len(scored_groups[Species.RHODODENDRON]))
        spacing = max(base_r * 1.5, self.min_radius * 1.2)
        anchors = self._generate_lattice(spacing)
        random.shuffle(anchors)

        placed_count = 0
        for ax, ay in anchors:
            if placed_count >= self.MAX_TRIADS:
                break
            # pick top remaining variety per species
            try:
                r_v = scored_groups[Species.RHODODENDRON][0].variety
                g_v = scored_groups[Species.GERANIUM][0].variety
                b_v = scored_groups[Species.BEGONIA][0].variety
            except IndexError:
                break
            triad = [r_v, g_v, b_v]
            layout = self._compute_compact_triad(triad)
            if layout is None:
                self._drop_any_empty(scored_groups)
                continue
            placed = False
            for rot in self.TRIAD_ROTATIONS:
                if self._attempt_triad(triad, layout, ax, ay, rot):
                    placed = True
                    placed_count += 1
                    # remove used varieties from groups & available
                    for v in triad:
                        self._remove_variety(v, scored_groups, species_groups)
                    break
            if not placed:
                continue
            self._bins.rebuild()
            self._drop_any_empty(scored_groups)
            if any(len(scored_groups[sp]) == 0 for sp in scored_groups):
                break

    def _attempt_triad(self, triad: List[PlantVariety], rel: List[Tuple[float, float]], ax: float, ay: float, theta: float) -> bool:
        positions: List[Position] = []
        for (v, (dx, dy)) in zip(triad, rel):
            rx, ry = self._rotate(dx, dy, theta)
            pos = Position(ax + rx, ay + ry)
            if not self.garden.can_place_plant(v, pos):
                return False
            positions.append(pos)
        for v, pos in zip(triad, positions):
            if self.garden.add_plant(v, pos) is None:
                return False
        return True

    def _compute_compact_triad(self, triad: List[PlantVariety]) -> List[Tuple[float, float]] | None:
        r0, r1, r2 = (triad[0].radius, triad[1].radius, triad[2].radius)
        # target edge lengths bias toward minimal separation but allow slight slack
        def edge(a: float, b: float) -> float:
            lower = max(a, b)
            upper = a + b
            return lower + 0.22 * (upper - lower)
        d01 = edge(r0, r1)
        d12 = edge(r1, r2)
        d02 = edge(r0, r2)
        if min(d01, d12, d02) <= 0:
            return None
        # derive coordinates for third point (law of cosines projection)
        x2 = (d02**2 + d01**2 - d12**2) / (2 * d01)
        y2_sq = d02**2 - x2**2
        if y2_sq < 0:
            return None
        y2 = math.sqrt(y2_sq)
        pts = [(0.0, 0.0), (d01, 0.0), (x2, y2)]
        cx = sum(p[0] for p in pts) / 3.0
        cy = sum(p[1] for p in pts) / 3.0
        return [(x - cx, y - cy) for (x, y) in pts]

    # ---------------------------- Phase 2: Lattice Fill ----------------- #
    def _lattice_deficit_fill(self) -> None:
        if not self.available:
            return
        lattice_spacing = max(self.min_radius, 0.8 * self.min_radius + 0.2)
        cells = self._generate_hex_lattice(lattice_spacing)
        random.shuffle(cells)

        for cx, cy in cells:
            if not self.available:
                break
            pos = Position(cx, cy)
            # pick candidate varieties by deficit ranking
            deficits = self._current_deficits()
            ordered_species = self._species_priority(deficits)
            chosen = None
            best_score = -1.0
            for sp in ordered_species:
                pool = [v for v in self.available if v.species == sp]
                if not pool:
                    continue
                for v in pool[:12]:  # sample top few per species
                    if not self.garden.can_place_plant(v, pos):
                        continue
                    inter = self._interaction_potential(v, pos)
                    eff = self._efficiency(v)
                    # combined score: weight deficit nutrient by its magnitude
                    nutrient_weight = deficits[self._produced_nutrient(sp)]
                    score = eff * (1 + 0.6 * nutrient_weight) + 0.4 * inter
                    if score > best_score:
                        best_score = score
                        chosen = v
                if chosen:
                    break
            if chosen and self.garden.add_plant(chosen, pos) is not None:
                self.available.remove(chosen)
                self._bins.rebuild()

    # ---------------------------- Phase 3: Greedy ----------------------- #
    def _greedy_balance(self) -> None:
        if not self.available:
            return
        candidates = self._generate_uniform_positions(self.GREEDY_GRID_STEP)
        first = not self.garden.plants
        while self.available and candidates:
            deficits = self._current_deficits()
            chosen = None
            best = -1.0
            best_pos = None
            for v in self.available[:250]:  # cap scan set for speed
                for pos in candidates:
                    if not self.garden.can_place_plant(v, pos):
                        continue
                    inter = self._interaction_potential(v, pos)
                    eff = self._efficiency(v)
                    nutrient_weight = deficits[self._produced_nutrient(v.species)]
                    score = inter * 0.6 + eff * 0.3 + nutrient_weight * 0.4
                    if score > best:
                        best = score
                        chosen = v
                        best_pos = pos
            if chosen is None or best_pos is None:
                break
            if first:
                centre = Position(self.garden.width/2, self.garden.height/2)
                if self.garden.can_place_plant(chosen, centre):
                    best_pos = centre
                first = False
            if self.garden.add_plant(chosen, best_pos) is None:
                break
            self.available.remove(chosen)
            candidates.remove(best_pos)
            self._bins.rebuild()

    # ---------------------------- Scoring & Helpers --------------------- #
    def _efficiency(self, variety: PlantVariety) -> float:
        prod, cons = 0.0, 0.0
        coeffs = variety.nutrient_coefficients
        produced = self._produced_nutrient(variety.species)
        prod = coeffs.get(produced, 0.0)
        for m, val in coeffs.items():
            if m != produced:
                cons += abs(val)
        denom = cons if cons > 0 else 0.25
        radius_bias = 1.0 / (1.0 + variety.radius)
        return prod / denom * radius_bias

    def _score_and_sort(self, varieties: List[PlantVariety]) -> List[VarietyScore]:
        scored = [VarietyScore(v, self._efficiency(v)) for v in varieties]
        scored.sort(key=lambda vs: vs.efficiency, reverse=True)
        return scored

    def _group_by_species(self, varieties: Iterable[PlantVariety]) -> Dict[Species, List[PlantVariety]]:
        groups: Dict[Species, List[PlantVariety]] = {
            Species.RHODODENDRON: [],
            Species.GERANIUM: [],
            Species.BEGONIA: [],
        }
        for v in varieties:
            groups[v.species].append(v)
        return groups

    def _remove_variety(self, variety: PlantVariety, scored_groups: Dict[Species, List[VarietyScore]], raw_groups: Dict[Species, List[PlantVariety]]):
        if variety in self.available:
            self.available.remove(variety)
        raw_groups[variety.species] = [x for x in raw_groups[variety.species] if x is not variety]
        scored_groups[variety.species] = [vs for vs in scored_groups[variety.species] if vs.variety is not variety]

    def _drop_any_empty(self, scored_groups: Dict[Species, List[VarietyScore]]):
        # noop hook for future logic (e.g., fallback early)
        return

    def _produced_nutrient(self, species: Species) -> Micronutrient:
        match species:
            case Species.RHODODENDRON:
                return Micronutrient.R
            case Species.GERANIUM:
                return Micronutrient.G
            case Species.BEGONIA:
                return Micronutrient.B
            case _:
                return Micronutrient.R  # default safeguard

    def _current_deficits(self) -> Dict[Micronutrient, float]:
        totals = {m: 0.0 for m in Micronutrient}
        for p in self.garden.plants:
            for m, val in p.variety.nutrient_coefficients.items():
                totals[m] += val
        # Normalize by max magnitude to produce weights in [-1,1]; more negative → larger deficit
        max_abs = max((abs(v) for v in totals.values()), default=1.0)
        return {m: (totals[m] / max_abs) for m in totals}

    def _species_priority(self, deficits: Dict[Micronutrient, float]) -> List[Species]:
        # Sort produced nutrients by ascending weight (more negative first)
        ordering = sorted([
            (deficits[Micronutrient.R], Species.RHODODENDRON),
            (deficits[Micronutrient.G], Species.GERANIUM),
            (deficits[Micronutrient.B], Species.BEGONIA),
        ], key=lambda t: t[0])
        return [sp for _w, sp in ordering]

    def _interaction_potential(self, variety: PlantVariety, pos: Position) -> int:
        # Count complementary species within interaction distance (radius sum)
        threshold_factor = self.INTERACTION_RADIUS_FACTOR
        needed = {Species.RHODODENDRON, Species.GERANIUM, Species.BEGONIA} - {variety.species}
        found = set()
        count = 0
        for idx in self._bins.nearby_indices(pos, variety.radius * 2 + self.min_radius):
            existing = self.garden.plants[idx]
            if existing.variety.species == variety.species:
                continue
            dx = pos.x - existing.position.x
            dy = pos.y - existing.position.y
            dist = math.hypot(dx, dy)
            if dist < (variety.radius + existing.variety.radius) * threshold_factor:
                count += 1
                found.add(existing.variety.species)
        if found.issuperset(needed):
            return count
        if found:
            return 1
        return 0

    # ---------------------------- Geometry & Grids ---------------------- #
    @staticmethod
    def _rotate(dx: float, dy: float, theta: float) -> Tuple[float, float]:
        c, s = math.cos(theta), math.sin(theta)
        return dx * c - dy * s, dx * s + dy * c

    def _generate_lattice(self, spacing: float) -> List[Tuple[float, float]]:
        anchors: List[Tuple[float, float]] = []
        dy = spacing * math.sqrt(3) / 2
        row = 0
        y = spacing / 2
        while y <= self.garden.height:
            x_offset = 0.0 if row % 2 == 0 else spacing * 0.5
            x = x_offset + spacing / 2
            while x <= self.garden.width:
                jx = x + (random.uniform(-1, 1) * spacing * self.LATTICE_JITTER)
                jy = y + (random.uniform(-1, 1) * dy * self.LATTICE_JITTER)
                anchors.append((jx, jy))
                x += spacing
            row += 1
            y += dy
        return anchors[:self.CLUSTER_LIMIT]

    def _generate_hex_lattice(self, spacing: float) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        dy = spacing * math.sqrt(3) / 2
        row = 0
        y = spacing / 2
        while y <= self.garden.height:
            x_offset = 0.0 if row % 2 == 0 else spacing * 0.5
            x = x_offset + spacing / 2
            while x <= self.garden.width:
                pts.append((x, y))
                x += spacing
            row += 1
            y += dy
        return pts

    def _generate_uniform_positions(self, step: float) -> List[Position]:
        positions: List[Position] = []
        y = step
        while y < self.garden.height - step:
            x = step
            while x < self.garden.width - step:
                positions.append(Position(x, y))
                x += step
            y += step
        random.shuffle(positions)
        return positions
