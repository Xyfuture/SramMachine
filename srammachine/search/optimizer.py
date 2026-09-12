"""Hypervolume-improvement simulated annealing for SplitTree candidates."""

from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
from typing import Optional, Tuple

from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from srammachine.mapping import (
    HardwareMapper, HardwareMappingRequest, HardwareMappingResult,
)
from srammachine.pipetree import PipeTree, TreeParser
from srammachine.simulator import SimulationResult, Simulator

from .config import SimulatedAnnealingConfig
from .hypervolume import (
    HypervolumeBounds, dominates, hypervolume_improvement,
)
from .mutations import legal_tree_mutations, rebind_split_tree
from .serialization import split_tree_to_dict


@dataclass(frozen=True)
class SplitTreeEvaluation:
    """One fully simulated workload/tree candidate."""

    global_batch_size: int
    mtp_enabled: bool
    split_tree: PipeTree
    mapping_result: HardwareMappingResult
    simulation_result: SimulationResult
    latency_ns: float
    single_user_throughput_per_second: float
    total_throughput_tokens_per_second: float
    first_seen_iteration: int

    def __post_init__(self) -> None:
        if type(self.global_batch_size) is not int or self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be a positive integer")
        if type(self.mtp_enabled) is not bool:
            raise TypeError("mtp_enabled must be a boolean")
        if not isinstance(self.split_tree, PipeTree):
            raise TypeError("split_tree must be a PipeTree")
        if not isinstance(self.mapping_result, HardwareMappingResult):
            raise TypeError("mapping_result must be a HardwareMappingResult")
        if not isinstance(self.simulation_result, SimulationResult):
            raise TypeError("simulation_result must be a SimulationResult")
        for name in (
            "latency_ns", "single_user_throughput_per_second",
            "total_throughput_tokens_per_second",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.first_seen_iteration) is not int or self.first_seen_iteration < 0:
            raise ValueError("first_seen_iteration must be a nonnegative integer")

    @property
    def objective_point(self) -> tuple[float, float]:
        return (
            self.single_user_throughput_per_second,
            self.total_throughput_tokens_per_second,
        )


@dataclass(frozen=True)
class ParetoPoint:
    """One raw objective point and all tied SplitTrees discovered for it."""

    single_user_throughput_per_second: float
    total_throughput_tokens_per_second: float
    normalized_single_user_throughput: float
    normalized_total_throughput: float
    latency_ns: float
    global_batch_size: int
    mtp_enabled: bool
    evaluations: Tuple[SplitTreeEvaluation, ...]

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        if not evaluations:
            raise ValueError("a ParetoPoint requires at least one evaluation")
        if any(not isinstance(item, SplitTreeEvaluation) for item in evaluations):
            raise TypeError("evaluations must contain SplitTreeEvaluation values")
        object.__setattr__(self, "evaluations", evaluations)

    @property
    def split_trees(self) -> Tuple[PipeTree, ...]:
        return tuple(evaluation.split_tree for evaluation in self.evaluations)

    @property
    def first_discovered_iteration(self) -> int:
        return min(item.first_seen_iteration for item in self.evaluations)


@dataclass(frozen=True)
class SplitTreeSearchResult:
    """Direct-run or annealing outcome."""

    enabled: bool
    initial_evaluation: SplitTreeEvaluation
    pareto_front: Tuple[ParetoPoint, ...]
    proposal_count: int
    simulation_count: int
    cache_hit_count: int
    accepted_proposal_count: int
    normalization_bounds: Optional[HypervolumeBounds]
    output_path: Optional[Path]

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if not isinstance(self.initial_evaluation, SplitTreeEvaluation):
            raise TypeError("initial_evaluation must be a SplitTreeEvaluation")
        front = tuple(self.pareto_front)
        if any(not isinstance(point, ParetoPoint) for point in front):
            raise TypeError("pareto_front must contain ParetoPoint values")
        for name in (
            "proposal_count", "simulation_count", "cache_hit_count",
            "accepted_proposal_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        object.__setattr__(self, "pareto_front", front)
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))


class SplitTreeOptimizer:
    """Run one SplitTree directly or search workload/tree candidates with SA."""

    def __init__(
        self, hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG,
    ) -> None:
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        self.hardware_config = hardware_config

    def run(
        self,
        request: HardwareMappingRequest,
        *,
        config: SimulatedAnnealingConfig = SimulatedAnnealingConfig(),
        initial_tree: Optional[PipeTree] = None,
    ) -> SplitTreeSearchResult:
        if not isinstance(request, HardwareMappingRequest):
            raise TypeError("request must be a HardwareMappingRequest")
        if not isinstance(config, SimulatedAnnealingConfig):
            raise TypeError("config must be a SimulatedAnnealingConfig")
        if initial_tree is not None and not isinstance(initial_tree, PipeTree):
            raise TypeError("initial_tree must be a PipeTree")

        self._request = request
        self._config = config
        self._mapper = HardwareMapper(self.hardware_config)
        self._parser = TreeParser(self.hardware_config)
        self._simulator = Simulator(self.hardware_config)
        self._mapping_cache = {}
        self._evaluation_cache = {}
        self._simulation_count = 0
        self._cache_hit_count = 0
        self._next_seen_iteration = 0

        if config.enabled:
            if request.inference_config.global_batch_size not in config.batch_sizes:
                raise ValueError(
                    "batch_sizes must contain the request's initial global batch"
                )
            self._batch_sizes = config.batch_sizes
            self._validate_workload_space()
        else:
            self._batch_sizes = (request.inference_config.global_batch_size,)

        initial_mapping = self._mapping(
            request.inference_config.global_batch_size,
            request.inference_config.mtp_enabled,
        )
        tree = initial_mapping.root_node if initial_tree is None else initial_tree
        self._validate_initial_tree(tree, initial_mapping)
        initial = self._evaluate(
            request.inference_config.global_batch_size,
            request.inference_config.mtp_enabled,
            tree,
        )

        if not config.enabled:
            point = self._pareto_points((initial,), None)
            return SplitTreeSearchResult(
                enabled=False,
                initial_evaluation=initial,
                pareto_front=point,
                proposal_count=0,
                simulation_count=self._simulation_count,
                cache_hit_count=self._cache_hit_count,
                accepted_proposal_count=0,
                normalization_bounds=None,
                output_path=None,
            )
        return self._anneal(initial)

    def _validate_workload_space(self) -> None:
        for batch in self._batch_sizes:
            for mtp in (False, True):
                try:
                    self._mapping(batch, mtp)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid annealing workload batch={batch}, mtp={mtp}: {error}"
                    ) from error

    @staticmethod
    def _validate_initial_tree(
        tree: PipeTree, mapping: HardwareMappingResult,
    ) -> None:
        if tree.batch_size != mapping.root_node.batch_size:
            raise ValueError("initial_tree batch_size does not match the request")
        if tree.operator_order != mapping.root_node.operator_order:
            raise ValueError("initial_tree operator_order does not match the request")

    def _mapping(self, batch: int, mtp: bool) -> HardwareMappingResult:
        key = (batch, mtp)
        cached = self._mapping_cache.get(key)
        if cached is not None:
            return cached
        inference = replace(
            self._request.inference_config,
            global_batch_size=batch,
            mtp_enabled=mtp,
        )
        mapping = self._mapper.map(HardwareMappingRequest(
            self._request.model_name, inference,
        ))
        self._mapping_cache[key] = mapping
        return mapping

    def _key(self, batch: int, mtp: bool, tree: PipeTree) -> str:
        return json.dumps(
            [batch, mtp, split_tree_to_dict(tree)],
            sort_keys=True,
            separators=(",", ":"),
        )

    def _evaluate(
        self, batch: int, mtp: bool, tree: PipeTree,
    ) -> SplitTreeEvaluation:
        key = self._key(batch, mtp, tree)
        cached = self._evaluation_cache.get(key)
        if cached is not None:
            self._cache_hit_count += 1
            return cached
        mapping = self._mapping(batch, mtp)
        graph = self._parser.parse(
            tree,
            mapping.operators,
            mapping.operator_mappings,
            layer_count=self._config.layer_count,
        )
        result = self._simulator.run(graph, mapping_result=mapping)
        latency_ns = result.latency_ns
        if latency_ns is None or latency_ns <= 0:
            raise ValueError("annealing candidate produced no positive model latency")
        accepted_tokens = mapping.request.inference_config.accepted_tokens_per_step
        single_user = accepted_tokens * 1_000_000_000 / latency_ns
        total = single_user * batch
        evaluation = SplitTreeEvaluation(
            global_batch_size=batch,
            mtp_enabled=mtp,
            split_tree=tree,
            mapping_result=mapping,
            simulation_result=result,
            latency_ns=latency_ns,
            single_user_throughput_per_second=single_user,
            total_throughput_tokens_per_second=total,
            first_seen_iteration=self._next_seen_iteration,
        )
        self._next_seen_iteration += 1
        self._simulation_count += 1
        self._evaluation_cache[key] = evaluation
        return evaluation

    def _workload_tree(
        self, tree: PipeTree, batch: int, mtp: bool,
    ) -> Optional[PipeTree]:
        mapping = self._mapping(batch, mtp)
        try:
            return rebind_split_tree(
                tree,
                batch_size=mapping.root_node.batch_size,
                operator_order=mapping.root_node.operator_order,
            )
        except (TypeError, ValueError):
            return None

    def _neighbors(
        self, evaluation: SplitTreeEvaluation,
    ) -> tuple[tuple[int, bool, PipeTree], ...]:
        batch = evaluation.global_batch_size
        mtp = evaluation.mtp_enabled
        tree = evaluation.split_tree
        candidates = [
            (batch, mtp, mutated)
            for _, mutated in legal_tree_mutations(tree)
        ]

        batch_index = self._batch_sizes.index(batch)
        for target_index in (batch_index - 1, batch_index + 1):
            if 0 <= target_index < len(self._batch_sizes):
                target_batch = self._batch_sizes[target_index]
                rebound = self._workload_tree(tree, target_batch, mtp)
                if rebound is not None:
                    candidates.append((target_batch, mtp, rebound))

        rebound = self._workload_tree(tree, batch, not mtp)
        if rebound is not None:
            candidates.append((batch, not mtp, rebound))

        unique = {}
        for candidate in candidates:
            unique.setdefault(self._key(*candidate), candidate)
        return tuple(unique.values())

    @staticmethod
    def _update_archive(
        archive: list[SplitTreeEvaluation], candidate: SplitTreeEvaluation,
    ) -> list[SplitTreeEvaluation]:
        if any(item is candidate for item in archive):
            return archive
        point = candidate.objective_point
        if any(dominates(item.objective_point, point) for item in archive):
            return archive
        return [
            item for item in archive
            if not dominates(point, item.objective_point)
        ] + [candidate]

    @staticmethod
    def _temperature(config: SimulatedAnnealingConfig, iteration: int) -> float:
        if config.rounds == 1:
            return float(config.initial_temperature)
        fraction = iteration / (config.rounds - 1)
        return float(config.initial_temperature) * (
            float(config.final_temperature) / float(config.initial_temperature)
        ) ** fraction

    @staticmethod
    def _domination_distance(
        archive: list[SplitTreeEvaluation],
        candidate: SplitTreeEvaluation,
        bounds: HypervolumeBounds,
    ) -> float:
        point = bounds.normalize(candidate.objective_point)
        dominators = [
            bounds.normalize(item.objective_point)
            for item in archive
            if dominates(item.objective_point, candidate.objective_point)
        ]
        comparison = dominators or [
            bounds.normalize(item.objective_point) for item in archive
        ]
        if not comparison:
            return 0.0
        return min(math.hypot(other[0] - point[0], other[1] - point[1])
                   for other in comparison)

    def _anneal(self, initial: SplitTreeEvaluation) -> SplitTreeSearchResult:
        rng = random.Random(self._config.random_seed)
        warmup = []
        for batch in self._batch_sizes:
            for mtp in (False, True):
                mapping = self._mapping(batch, mtp)
                warmup.append(self._evaluate(batch, mtp, mapping.root_node))

        warm_current = initial
        for _ in range(self._config.warmup_rounds):
            neighbors = self._neighbors(warm_current)
            if not neighbors:
                raise ValueError("annealing candidate has no legal neighbors")
            warm_current = self._evaluate(*rng.choice(neighbors))
            warmup.append(warm_current)

        bounds = HypervolumeBounds.from_points(
            item.objective_point for item in warmup
        )
        archive = []
        for evaluation in warmup:
            archive = self._update_archive(archive, evaluation)

        current = initial
        accepted = 0
        for iteration in range(self._config.rounds):
            neighbors = self._neighbors(current)
            if not neighbors:
                raise ValueError("annealing candidate has no legal neighbors")
            candidate = self._evaluate(*rng.choice(neighbors))
            normalized_archive = tuple(
                bounds.normalize(item.objective_point) for item in archive
            )
            normalized_candidate = bounds.normalize(candidate.objective_point)
            reward = hypervolume_improvement(
                normalized_archive, normalized_candidate,
            )
            distance = self._domination_distance(archive, candidate, bounds)
            archive = self._update_archive(archive, candidate)
            temperature = self._temperature(self._config, iteration)
            if reward > 0 or rng.random() < math.exp(-distance / temperature):
                current = candidate
                accepted += 1

        front = self._pareto_points(tuple(archive), bounds)
        output_path = self._write_result(front, bounds, accepted)
        return SplitTreeSearchResult(
            enabled=True,
            initial_evaluation=initial,
            pareto_front=front,
            proposal_count=self._config.rounds,
            simulation_count=self._simulation_count,
            cache_hit_count=self._cache_hit_count,
            accepted_proposal_count=accepted,
            normalization_bounds=bounds,
            output_path=output_path,
        )

    @staticmethod
    def _pareto_points(
        archive: tuple[SplitTreeEvaluation, ...],
        bounds: Optional[HypervolumeBounds],
    ) -> Tuple[ParetoPoint, ...]:
        grouped = {}
        for evaluation in archive:
            key = (
                evaluation.objective_point,
                evaluation.global_batch_size,
                evaluation.mtp_enabled,
                evaluation.latency_ns,
            )
            grouped.setdefault(key, []).append(evaluation)
        points = []
        for (
            (single_user, total), batch, mtp, latency_ns,
        ), evaluations in grouped.items():
            normalized = (1.0, 1.0) if bounds is None else bounds.normalize(
                (single_user, total)
            )
            points.append(ParetoPoint(
                single_user_throughput_per_second=single_user,
                total_throughput_tokens_per_second=total,
                normalized_single_user_throughput=normalized[0],
                normalized_total_throughput=normalized[1],
                latency_ns=latency_ns,
                global_batch_size=batch,
                mtp_enabled=mtp,
                evaluations=tuple(sorted(
                    evaluations,
                    key=lambda item: item.first_seen_iteration,
                )),
            ))
        return tuple(sorted(
            points,
            key=lambda point: (
                point.single_user_throughput_per_second,
                -point.total_throughput_tokens_per_second,
            ),
        ))

    def _write_result(
        self,
        front: Tuple[ParetoPoint, ...],
        bounds: HypervolumeBounds,
        accepted: int,
    ) -> Path:
        inference = self._request.inference_config
        payload = {
            "format_version": 1,
            "reference_point": [0.0, 0.0],
            "model": self._request.model_name,
            "hardware": {"chip_count": self.hardware_config.chip_count},
            "inference": {
                "input_sequence_length": inference.input_sequence_length,
                "output_sequence_length": inference.output_sequence_length,
                "kv_cache_dtype": inference.kv_cache_dtype,
                "moe_parallel_strategy": inference.moe_parallel_strategy.value,
            },
            "annealing": {
                "batch_sizes": list(self._batch_sizes),
                "mtp_values": [False, True],
                "warmup_rounds": self._config.warmup_rounds,
                "rounds": self._config.rounds,
                "random_seed": self._config.random_seed,
                "initial_temperature": self._config.initial_temperature,
                "final_temperature": self._config.final_temperature,
                "layer_count": self._config.layer_count,
                "proposal_count": self._config.rounds,
                "simulation_count": self._simulation_count,
                "cache_hit_count": self._cache_hit_count,
                "accepted_proposal_count": accepted,
            },
            "normalization_bounds": {
                "x_min": bounds.x_min,
                "x_max": bounds.x_max,
                "y_min": bounds.y_min,
                "y_max": bounds.y_max,
            },
            "pareto_front": [self._point_payload(point) for point in front],
        }
        output_dir = self._config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M")
        model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._request.model_name)
        strategy = inference.moe_parallel_strategy.value
        batch_range = f"{self._batch_sizes[0]}-{self._batch_sizes[-1]}"
        stem = (
            f"{timestamp}_{model}_bs{batch_range}_{strategy}_"
            "mtp_search_sa_pareto"
        )
        suffix = 1
        while True:
            postfix = "" if suffix == 1 else f"_{suffix:02d}"
            path = output_dir / f"{stem}{postfix}.json"
            try:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                return path
            except FileExistsError:
                suffix += 1

    @staticmethod
    def _point_payload(point: ParetoPoint) -> dict:
        return {
            "single_user_throughput_per_second": (
                point.single_user_throughput_per_second
            ),
            "total_throughput_tokens_per_second": (
                point.total_throughput_tokens_per_second
            ),
            "normalized_single_user_throughput": (
                point.normalized_single_user_throughput
            ),
            "normalized_total_throughput": point.normalized_total_throughput,
            "latency_ns": point.latency_ns,
            "global_batch_size": point.global_batch_size,
            "mtp_enabled": point.mtp_enabled,
            "first_discovered_iteration": point.first_discovered_iteration,
            "split_trees": [
                split_tree_to_dict(tree) for tree in point.split_trees
            ],
        }


__all__ = [
    "SplitTreeEvaluation", "ParetoPoint", "SplitTreeSearchResult",
    "SplitTreeOptimizer",
]
