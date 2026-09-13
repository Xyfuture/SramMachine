"""Per-workload hypervolume-improvement annealing for SplitTrees."""

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

from .config import (
    SimulatedAnnealingConfig, derive_restart_seed, derive_workload_seed,
)
from .hypervolume import HypervolumeBounds, dominates, hypervolume_improvement
from .mutations import legal_tree_mutations, rebind_split_tree
from .serialization import split_tree_to_dict


def _select_proposal(candidates, rng, score_cache, tree_key, parse):
    """Prefer a valid, globally unevaluated neighbor over cached neighbors."""
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    cached = []
    for tree in shuffled:
        key = tree_key(tree)
        if key in score_cache:
            cached.append(tree)
            continue
        graph = parse(tree)
        if graph is not None:
            return tree, graph
    if cached:
        return cached[0], None
    raise ValueError("annealing candidate has no legal neighbors")


@dataclass(frozen=True)
class SplitTreeEvaluation:
    """One fully materialized workload/tree simulation."""

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
class _CandidateScore:
    """Lightweight internal score; deliberately excludes graph/trace objects."""

    global_batch_size: int
    mtp_enabled: bool
    split_tree: PipeTree
    latency_ns: float
    single_user_throughput_per_second: float
    total_throughput_tokens_per_second: float
    first_seen_iteration: int

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
class AnnealingRestartResult:
    """Search statistics for one deterministic annealing restart."""

    restart_index: int
    random_seed: int
    proposal_count: int
    candidate_simulation_count: int
    cache_hit_count: int
    accepted_proposal_count: int
    normalization_bounds: HypervolumeBounds

    def __post_init__(self) -> None:
        if type(self.restart_index) is not int or self.restart_index < 0:
            raise ValueError("restart_index must be a nonnegative integer")
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer")
        for name in (
            "proposal_count", "candidate_simulation_count", "cache_hit_count",
            "accepted_proposal_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.normalization_bounds, HypervolumeBounds):
            raise TypeError("normalization_bounds must be HypervolumeBounds")


@dataclass(frozen=True)
class WorkloadSearchResult:
    """Independent SplitTree search result for one fixed batch/MTP workload."""

    global_batch_size: int
    mtp_enabled: bool
    baseline_evaluation: SplitTreeEvaluation
    initial_evaluation: SplitTreeEvaluation
    best_evaluations: Tuple[SplitTreeEvaluation, ...]
    proposal_count: int
    candidate_simulation_count: int
    materialization_simulation_count: int
    cache_hit_count: int
    accepted_proposal_count: int
    normalization_bounds: HypervolumeBounds
    initial_tree_fallback: bool
    random_seed: int = 20260912
    restart_results: Tuple[AnnealingRestartResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer")
        restarts = tuple(self.restart_results)
        if not restarts or any(
            not isinstance(item, AnnealingRestartResult) for item in restarts
        ):
            raise ValueError("restart_results must contain restart statistics")
        if tuple(item.restart_index for item in restarts) != tuple(range(len(restarts))):
            raise ValueError("restart indices must be contiguous from zero")
        for aggregate_name in (
            "proposal_count", "candidate_simulation_count", "cache_hit_count",
            "accepted_proposal_count",
        ):
            if getattr(self, aggregate_name) != sum(
                getattr(item, aggregate_name) for item in restarts
            ):
                raise ValueError(
                    f"{aggregate_name} must equal the sum across restarts"
                )
        best = tuple(self.best_evaluations)
        if not best or any(not isinstance(item, SplitTreeEvaluation) for item in best):
            raise ValueError("best_evaluations must contain evaluations")
        object.__setattr__(self, "best_evaluations", best)
        object.__setattr__(self, "restart_results", restarts)

    @property
    def restart_count(self) -> int:
        return len(self.restart_results)

    @property
    def best_evaluation(self) -> SplitTreeEvaluation:
        return self.best_evaluations[0]

    @property
    def latency_reduction_fraction(self) -> float:
        return 1.0 - (
            self.best_evaluation.latency_ns / self.baseline_evaluation.latency_ns
        )

    @property
    def throughput_improvement_fraction(self) -> float:
        return (
            self.best_evaluation.total_throughput_tokens_per_second
            / self.baseline_evaluation.total_throughput_tokens_per_second
            - 1.0
        )

    @property
    def simulator_run_count(self) -> int:
        return self.candidate_simulation_count + self.materialization_simulation_count


@dataclass(frozen=True)
class SplitTreeSearchResult:
    """Direct-run or independent per-workload annealing outcome."""

    enabled: bool
    initial_evaluation: SplitTreeEvaluation
    pareto_front: Tuple[ParetoPoint, ...]
    workload_results: Tuple[WorkloadSearchResult, ...]
    proposal_count: int
    simulation_count: int
    candidate_simulation_count: int
    materialization_simulation_count: int
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
        workloads = tuple(self.workload_results)
        if any(not isinstance(point, ParetoPoint) for point in front):
            raise TypeError("pareto_front must contain ParetoPoint values")
        if any(not isinstance(item, WorkloadSearchResult) for item in workloads):
            raise TypeError("workload_results must contain WorkloadSearchResult values")
        for name in (
            "proposal_count", "simulation_count", "candidate_simulation_count",
            "materialization_simulation_count", "cache_hit_count",
            "accepted_proposal_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.simulation_count != (
            self.candidate_simulation_count + self.materialization_simulation_count
        ):
            raise ValueError("simulation_count must include every Simulator.run call")
        object.__setattr__(self, "pareto_front", front)
        object.__setattr__(self, "workload_results", workloads)
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))


class SplitTreeOptimizer:
    """Run one tree directly or search every batch/MTP workload fairly."""

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
        self._all_objective_points = []
        self._next_seen_iteration = 0
        self._total_candidate_simulations = 0
        self._total_materializations = 0
        self._total_cache_hits = 0

        request_mapping = self._mapping(
            request.inference_config.global_batch_size,
            request.inference_config.mtp_enabled,
        )
        tree = request_mapping.root_node if initial_tree is None else initial_tree
        self._validate_initial_tree(tree, request_mapping)

        if not config.enabled:
            evaluation = self._simulate_full(request_mapping, tree)
            front = self._pareto_points((evaluation,), None)
            return SplitTreeSearchResult(
                False, evaluation, front, (), 0, 1, 1, 0, 0, 0,
                None, None,
            )

        if request.inference_config.global_batch_size not in config.batch_sizes:
            raise ValueError("batch_sizes must contain the request's initial global batch")
        if request.inference_config.mtp_enabled not in config.mtp_values:
            raise ValueError("mtp_values must contain the request's initial MTP value")
        self._validate_workload_space()

        workload_results = []
        for batch in config.batch_sizes:
            for mtp in config.mtp_values:
                seed = derive_workload_seed(
                    config.random_seed,
                    request.model_name,
                    request.inference_config.moe_parallel_strategy.value,
                    batch,
                    mtp,
                )
                workload_results.append(self._search_workload(
                    batch, mtp, initial_tree, seed,
                ))

        global_bounds = HypervolumeBounds.from_points(self._all_objective_points)
        global_archive = []
        for workload in workload_results:
            for evaluation in workload.best_evaluations:
                global_archive = self._update_archive(global_archive, evaluation)
        front = self._pareto_points(tuple(global_archive), global_bounds)
        initial_workload = next(
            item for item in workload_results
            if item.global_batch_size == request.inference_config.global_batch_size
            and item.mtp_enabled is request.inference_config.mtp_enabled
        )
        output_path = (
            self._write_result(tuple(workload_results), front, global_bounds)
            if config.write_result else None
        )
        proposals = sum(item.proposal_count for item in workload_results)
        accepted = sum(item.accepted_proposal_count for item in workload_results)
        return SplitTreeSearchResult(
            True, initial_workload.initial_evaluation, front,
            tuple(workload_results), proposals,
            self._total_candidate_simulations + self._total_materializations,
            self._total_candidate_simulations, self._total_materializations,
            self._total_cache_hits, accepted, global_bounds, output_path,
        )

    def _validate_workload_space(self) -> None:
        for batch in self._config.batch_sizes:
            for mtp in self._config.mtp_values:
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
        if key not in self._mapping_cache:
            inference = replace(
                self._request.inference_config,
                global_batch_size=batch,
                mtp_enabled=mtp,
            )
            self._mapping_cache[key] = self._mapper.map(HardwareMappingRequest(
                self._request.model_name, inference,
            ))
        return self._mapping_cache[key]

    @staticmethod
    def _tree_key(tree: PipeTree) -> str:
        return json.dumps(
            split_tree_to_dict(tree), sort_keys=True, separators=(",", ":"),
        )

    @staticmethod
    def _metrics(mapping: HardwareMappingResult, result: SimulationResult):
        latency_ns = result.latency_ns
        if latency_ns is None or latency_ns <= 0:
            raise ValueError("annealing candidate produced no positive model latency")
        accepted = mapping.request.inference_config.accepted_tokens_per_step
        single_user = accepted * 1_000_000_000 / latency_ns
        batch = mapping.request.inference_config.global_batch_size
        return latency_ns, single_user, single_user * batch

    def _new_score(
        self, mapping: HardwareMappingResult, tree: PipeTree,
        result: SimulationResult,
    ) -> _CandidateScore:
        latency, single_user, total = self._metrics(mapping, result)
        score = _CandidateScore(
            mapping.request.inference_config.global_batch_size,
            mapping.request.inference_config.mtp_enabled,
            tree, latency, single_user, total, self._next_seen_iteration,
        )
        self._next_seen_iteration += 1
        self._all_objective_points.append(score.objective_point)
        return score

    def _evaluation(
        self, mapping: HardwareMappingResult, score: _CandidateScore,
        result: SimulationResult,
    ) -> SplitTreeEvaluation:
        return SplitTreeEvaluation(
            score.global_batch_size, score.mtp_enabled, score.split_tree,
            mapping, result, score.latency_ns,
            score.single_user_throughput_per_second,
            score.total_throughput_tokens_per_second,
            score.first_seen_iteration,
        )

    def _simulate_full(
        self, mapping: HardwareMappingResult, tree: PipeTree,
    ) -> SplitTreeEvaluation:
        graph = self._parser.parse(
            tree, mapping.operators, mapping.operator_mappings,
            layer_count=self._config.layer_count,
        )
        result = self._simulator.run(graph, mapping_result=mapping)
        score = self._new_score(mapping, tree, result)
        self._total_candidate_simulations += 1
        return self._evaluation(mapping, score, result)

    def _search_workload(
        self, batch: int, mtp: bool, initial_tree: Optional[PipeTree], seed: int,
    ) -> WorkloadSearchResult:
        mapping = self._mapping(batch, mtp)
        score_cache = {}
        full_cache = {}
        invalid_keys = set()
        candidate_simulations = 0
        materializations = 0
        cache_hits = 0

        def parse(tree: PipeTree):
            key = self._tree_key(tree)
            if key in invalid_keys:
                return None
            try:
                return self._parser.parse(
                    tree, mapping.operators, mapping.operator_mappings,
                    layer_count=self._config.layer_count,
                )
            except ValueError:
                invalid_keys.add(key)
                return None

        def score(
            tree: PipeTree, graph=None, *, retain_full: bool = False,
        ) -> _CandidateScore:
            nonlocal candidate_simulations, cache_hits
            key = self._tree_key(tree)
            if key in score_cache:
                cache_hits += 1
                return score_cache[key]
            graph = parse(tree) if graph is None else graph
            if graph is None:
                raise ValueError("SplitTree cannot be lowered exactly for this workload")
            result = self._simulator.run(graph, mapping_result=mapping)
            item = self._new_score(mapping, tree, result)
            score_cache[key] = item
            if retain_full:
                full_cache[key] = self._evaluation(mapping, item, result)
            candidate_simulations += 1
            return item

        def materialize(item: _CandidateScore) -> SplitTreeEvaluation:
            nonlocal materializations
            key = self._tree_key(item.split_tree)
            if key in full_cache:
                return full_cache[key]
            graph = parse(item.split_tree)
            if graph is None:
                raise RuntimeError("previously simulated SplitTree became invalid")
            result = self._simulator.run(graph, mapping_result=mapping)
            if self._metrics(mapping, result) != (
                item.latency_ns,
                item.single_user_throughput_per_second,
                item.total_throughput_tokens_per_second,
            ):
                raise RuntimeError("materialized simulation does not match cached score")
            evaluation = self._evaluation(mapping, item, result)
            full_cache[key] = evaluation
            materializations += 1
            return evaluation

        baseline_tree = mapping.root_node
        baseline_graph = parse(baseline_tree)
        if baseline_graph is None:
            raise RuntimeError("mapper baseline RootNode cannot be parsed")
        baseline_score = score(
            baseline_tree, baseline_graph, retain_full=True,
        )

        fallback = False
        start_tree = baseline_tree
        if initial_tree is not None:
            try:
                rebound = rebind_split_tree(
                    initial_tree,
                    batch_size=baseline_tree.batch_size,
                    operator_order=baseline_tree.operator_order,
                )
            except (TypeError, ValueError):
                fallback = True
            else:
                if parse(rebound) is None:
                    fallback = True
                else:
                    start_tree = rebound
        start_score = score(
            start_tree, retain_full=(start_tree != baseline_tree),
        )

        def proposal(current: _CandidateScore, rng: random.Random):
            return _select_proposal(
                (tree for _, tree in legal_tree_mutations(current.split_tree)),
                rng, score_cache, self._tree_key, parse,
            )

        combined_archive = []
        all_warmup = []
        restart_results = []
        setup_simulations = candidate_simulations
        setup_cache_hits = cache_hits
        for restart_index in range(self._config.restart_count):
            restart_seed = derive_restart_seed(seed, restart_index)
            rng = random.Random(restart_seed)
            simulations_before = candidate_simulations
            hits_before = cache_hits

            warmup = [baseline_score, start_score]
            warm_current = start_score
            for _ in range(self._config.warmup_rounds):
                tree, graph = proposal(warm_current, rng)
                warm_current = score(tree, graph)
                warmup.append(warm_current)
            all_warmup.extend(warmup)
            bounds = HypervolumeBounds.from_points(
                item.objective_point for item in warmup
            )
            archive = []
            for item in warmup:
                archive = self._update_archive(archive, item)

            # Continue from this restart's warmup endpoint. Each restart owns
            # its state, bounds, temperature, and archive; only expensive
            # simulation scores are shared across restarts.
            current = warm_current
            accepted = 0
            for iteration in range(self._config.rounds):
                tree, graph = proposal(current, rng)
                candidate = score(tree, graph)
                normalized_archive = tuple(
                    bounds.normalize(item.objective_point) for item in archive
                )
                reward = hypervolume_improvement(
                    normalized_archive, bounds.normalize(candidate.objective_point),
                )
                distance = self._domination_distance(archive, candidate, bounds)
                archive = self._update_archive(archive, candidate)
                temperature = self._temperature(self._config, iteration)
                if reward > 0 or rng.random() < math.exp(-distance / temperature):
                    current = candidate
                    accepted += 1

            for item in archive:
                combined_archive = self._update_archive(combined_archive, item)
            restart_results.append(AnnealingRestartResult(
                restart_index=restart_index,
                random_seed=restart_seed,
                proposal_count=self._config.rounds,
                candidate_simulation_count=(
                    candidate_simulations - simulations_before
                    + (setup_simulations if restart_index == 0 else 0)
                ),
                cache_hit_count=(
                    cache_hits - hits_before
                    + (setup_cache_hits if restart_index == 0 else 0)
                ),
                accepted_proposal_count=accepted,
                normalization_bounds=bounds,
            ))

        workload_bounds = HypervolumeBounds.from_points(
            item.objective_point for item in all_warmup
        )
        best_point = max(item.objective_point for item in combined_archive)
        best_scores = tuple(
            item for item in combined_archive if item.objective_point == best_point
        )
        baseline_evaluation = materialize(baseline_score)
        initial_evaluation = materialize(start_score)
        best_evaluations = tuple(materialize(item) for item in best_scores)

        self._total_candidate_simulations += candidate_simulations
        self._total_materializations += materializations
        self._total_cache_hits += cache_hits
        return WorkloadSearchResult(
            global_batch_size=batch,
            mtp_enabled=mtp,
            baseline_evaluation=baseline_evaluation,
            initial_evaluation=initial_evaluation,
            best_evaluations=best_evaluations,
            proposal_count=self._config.rounds * self._config.restart_count,
            candidate_simulation_count=candidate_simulations,
            materialization_simulation_count=materializations,
            cache_hit_count=cache_hits,
            accepted_proposal_count=sum(
                item.accepted_proposal_count for item in restart_results
            ),
            normalization_bounds=workload_bounds,
            initial_tree_fallback=fallback,
            random_seed=seed,
            restart_results=tuple(restart_results),
        )

    @staticmethod
    def _update_archive(archive: list, candidate):
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
    def _domination_distance(archive, candidate, bounds: HypervolumeBounds) -> float:
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
        for ((single_user, total), batch, mtp, latency), evaluations in grouped.items():
            normalized = (1.0, 1.0) if bounds is None else bounds.normalize(
                (single_user, total)
            )
            points.append(ParetoPoint(
                single_user, total, normalized[0], normalized[1], latency,
                batch, mtp, tuple(sorted(
                    evaluations, key=lambda item: item.first_seen_iteration,
                )),
            ))
        return tuple(sorted(points, key=lambda point: (
            point.single_user_throughput_per_second,
            -point.total_throughput_tokens_per_second,
        )))

    def _write_result(
        self, workloads: Tuple[WorkloadSearchResult, ...],
        front: Tuple[ParetoPoint, ...], bounds: HypervolumeBounds,
    ) -> Path:
        inference = self._request.inference_config
        payload = {
            "format_version": 2,
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
                "batch_sizes": list(self._config.batch_sizes),
                "mtp_values": list(self._config.mtp_values),
                "warmup_rounds_per_workload": self._config.warmup_rounds,
                "rounds_per_workload": self._config.rounds,
                "restart_count_per_workload": self._config.restart_count,
                "random_seed": self._config.random_seed,
                "initial_temperature": self._config.initial_temperature,
                "final_temperature": self._config.final_temperature,
                "layer_count": self._config.layer_count,
                "proposal_count": sum(item.proposal_count for item in workloads),
                "candidate_simulation_count": self._total_candidate_simulations,
                "materialization_simulation_count": self._total_materializations,
                "simulation_count": (
                    self._total_candidate_simulations + self._total_materializations
                ),
                "cache_hit_count": self._total_cache_hits,
                "accepted_proposal_count": sum(
                    item.accepted_proposal_count for item in workloads
                ),
            },
            "normalization_bounds": self._bounds_payload(bounds),
            "workloads": [self._workload_payload(item) for item in workloads],
            "pareto_front": [self._point_payload(point) for point in front],
        }
        output_dir = self._config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M")
        model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._request.model_name)
        strategy = inference.moe_parallel_strategy.value
        batches = self._config.batch_sizes
        batch_range = f"{batches[0]}-{batches[-1]}"
        stem = f"{timestamp}_{model}_bs{batch_range}_{strategy}_mtp_search_sa_pareto"
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
    def _bounds_payload(bounds: HypervolumeBounds) -> dict:
        return {
            "x_min": bounds.x_min, "x_max": bounds.x_max,
            "y_min": bounds.y_min, "y_max": bounds.y_max,
        }

    @classmethod
    def _workload_payload(cls, item: WorkloadSearchResult) -> dict:
        baseline = item.baseline_evaluation
        best = item.best_evaluation
        return {
            "global_batch_size": item.global_batch_size,
            "mtp_enabled": item.mtp_enabled,
            "random_seed": item.random_seed,
            "restart_count": item.restart_count,
            "restart_results": [
                {
                    "restart_index": restart.restart_index,
                    "random_seed": restart.random_seed,
                    "proposal_count": restart.proposal_count,
                    "candidate_simulation_count": (
                        restart.candidate_simulation_count
                    ),
                    "cache_hit_count": restart.cache_hit_count,
                    "accepted_proposal_count": restart.accepted_proposal_count,
                    "normalization_bounds": cls._bounds_payload(
                        restart.normalization_bounds
                    ),
                }
                for restart in item.restart_results
            ],
            "baseline_latency_ns": baseline.latency_ns,
            "best_latency_ns": best.latency_ns,
            "latency_reduction_fraction": item.latency_reduction_fraction,
            "baseline_total_throughput_tokens_per_second": (
                baseline.total_throughput_tokens_per_second
            ),
            "best_total_throughput_tokens_per_second": (
                best.total_throughput_tokens_per_second
            ),
            "throughput_improvement_fraction": item.throughput_improvement_fraction,
            "proposal_count": item.proposal_count,
            "candidate_simulation_count": item.candidate_simulation_count,
            "materialization_simulation_count": item.materialization_simulation_count,
            "simulator_run_count": item.simulator_run_count,
            "cache_hit_count": item.cache_hit_count,
            "accepted_proposal_count": item.accepted_proposal_count,
            "initial_tree_fallback": item.initial_tree_fallback,
            "normalization_bounds": cls._bounds_payload(item.normalization_bounds),
            "best_split_trees": [
                split_tree_to_dict(evaluation.split_tree)
                for evaluation in item.best_evaluations
            ],
        }

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
            "split_trees": [split_tree_to_dict(tree) for tree in point.split_trees],
        }


__all__ = [
    "SplitTreeEvaluation", "ParetoPoint", "AnnealingRestartResult",
    "WorkloadSearchResult",
    "SplitTreeSearchResult", "SplitTreeOptimizer",
]
