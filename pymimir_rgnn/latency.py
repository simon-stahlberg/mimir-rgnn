import statistics
import time
import torch

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Sequence

from .model import RelationalGraphNeuralNetwork


BenchmarkMode = Literal['inference', 'training']
TimerKind = Literal['wall_clock', 'cuda_event']


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics for a latency measurement in milliseconds."""

    num_iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    @staticmethod
    def from_samples(samples_ms: Sequence[float]) -> 'LatencyStats':
        """Build latency statistics from a sequence of millisecond samples.

        Args:
            samples_ms: Measured latencies in milliseconds.

        Returns:
            Summary statistics for the provided samples.
        """
        assert len(samples_ms) > 0, 'At least one latency sample is required.'
        sorted_samples = sorted(samples_ms)
        p95_index = min(len(sorted_samples) - 1, int(0.95 * (len(sorted_samples) - 1)))
        stdev = statistics.stdev(sorted_samples) if len(sorted_samples) > 1 else 0.0
        return LatencyStats(
            num_iterations=len(sorted_samples),
            mean_ms=statistics.mean(sorted_samples),
            median_ms=statistics.median(sorted_samples),
            p95_ms=sorted_samples[p95_index],
            min_ms=sorted_samples[0],
            max_ms=sorted_samples[-1],
            stdev_ms=stdev,
        )

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation of the statistics."""
        return asdict(self)


@dataclass(frozen=True)
class ForwardReadoutLatency:
    """Latency breakdown for the model.forward(...).readout(...) path."""

    mode: BenchmarkMode
    device_type: str
    device_name: str | None
    readout_names: tuple[str, ...]
    warmup_iterations: int
    trials: int
    timers: dict[str, TimerKind]
    total: LatencyStats
    encode: LatencyStats
    compute: LatencyStats
    readout: LatencyStats

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the latency report."""
        return {
            'mode': self.mode,
            'device_type': self.device_type,
            'device_name': self.device_name,
            'readout_names': self.readout_names,
            'warmup_iterations': self.warmup_iterations,
            'trials': self.trials,
            'timers': self.timers,
            'total': self.total.to_dict(),
            'encode': self.encode.to_dict(),
            'compute': self.compute.to_dict(),
            'readout': self.readout.to_dict(),
        }


def _normalize_readout_names(readout_names: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(readout_names, str):
        return (readout_names,)
    normalized = tuple(readout_names)
    assert len(normalized) > 0, 'At least one readout name is required.'
    return normalized


def _synchronize(device: torch.device, synchronize: bool) -> None:
    if synchronize and device.type == 'cuda':
        torch.cuda.synchronize(device)


def _context_for_mode(mode: BenchmarkMode) -> Callable[[], Any]:
    return torch.inference_mode if mode == 'inference' else torch.enable_grad


def _resolve_section_timer(device: torch.device, use_cuda_events: bool) -> TimerKind:
    return 'cuda_event' if use_cuda_events and device.type == 'cuda' else 'wall_clock'


def _measure_operation_ms(
    operation: Callable[[], Any],
    *,
    device: torch.device,
    synchronize: bool,
    timer: TimerKind,
) -> float:
    if timer == 'cuda_event':
        assert device.type == 'cuda', 'CUDA event timing requires a CUDA device.'
        _synchronize(device, synchronize)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        operation()
        end_event.record()
        end_event.synchronize()
        return float(start_event.elapsed_time(end_event))

    _synchronize(device, synchronize)
    start = time.perf_counter()
    operation()
    _synchronize(device, synchronize)
    return (time.perf_counter() - start) * 1000.0


def _measure_samples(
    operation: Callable[[], Any],
    *,
    iterations: int,
    warmup_iterations: int,
    trials: int,
    mode: BenchmarkMode,
    device: torch.device,
    synchronize: bool,
    timer: TimerKind,
) -> tuple[float, ...]:
    context_factory = _context_for_mode(mode)
    with context_factory():
        samples_ms: list[float] = []
        for _ in range(trials):
            for _ in range(warmup_iterations):
                operation()
            for _ in range(iterations):
                samples_ms.append(
                    _measure_operation_ms(
                        operation,
                        device=device,
                        synchronize=synchronize,
                        timer=timer,
                    )
                )
    return tuple(samples_ms)


def _measure_prepared_samples(
    prepare: Callable[[], Any],
    operation: Callable[[Any], Any],
    *,
    iterations: int,
    warmup_iterations: int,
    trials: int,
    mode: BenchmarkMode,
    device: torch.device,
    synchronize: bool,
    timer: TimerKind,
) -> tuple[float, ...]:
    context_factory = _context_for_mode(mode)
    with context_factory():
        samples_ms: list[float] = []
        for _ in range(trials):
            for _ in range(warmup_iterations):
                prepared = prepare()
                operation(prepared)
            for _ in range(iterations):
                prepared = prepare()
                samples_ms.append(
                    _measure_operation_ms(
                        lambda: operation(prepared),
                        device=device,
                        synchronize=synchronize,
                        timer=timer,
                    )
                )
    return tuple(samples_ms)


def _run_readouts(forward_state: Any, readout_names: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(forward_state.readout(name) for name in readout_names)


def measure_forward_readout_latency(
    model: RelationalGraphNeuralNetwork,
    x: list[tuple],
    readout_names: str | Sequence[str],
    *,
    iterations: int = 100,
    warmup_iterations: int = 20,
    trials: int = 1,
    mode: BenchmarkMode = 'inference',
    synchronize: bool | None = None,
    use_cuda_events: bool | None = None,
) -> ForwardReadoutLatency:
    """Measure latency of the end-to-end forward and readout path.

    The benchmark records the direct latency of `model.forward(...).readout(...)`
    and a phase breakdown using `curry_forward()` for the encode and compute
    portions of the path.

    Args:
        model: The R-GNN model to benchmark.
        x: The list of input tuples for the model.
        readout_names: One or more readout names to evaluate after the forward pass.
        iterations: Number of measured iterations.
        warmup_iterations: Number of warmup iterations to run before timing.
        trials: Number of repeated benchmark trials. Each trial gets its own warmup.
        mode: Whether to benchmark inference or training-style forward passes.
        synchronize: Whether to synchronize the device before and after each timed section.
            Defaults to True on CUDA and False otherwise.
        use_cuda_events: Whether to use CUDA events for the compute and readout sections
            on CUDA devices. End-to-end and encode timing remain synchronized wall-clock
            so they include Python and host-side overhead.

    Returns:
        Summary statistics for total, encode, compute, and readout latencies.
    """
    assert iterations > 0, 'iterations must be positive.'
    assert warmup_iterations >= 0, 'warmup_iterations must be non-negative.'
    assert trials > 0, 'trials must be positive.'
    normalized_readout_names = _normalize_readout_names(readout_names)
    device = model.get_device()
    should_synchronize = device.type == 'cuda' if synchronize is None else synchronize
    should_use_cuda_events = device.type == 'cuda' if use_cuda_events is None else use_cuda_events
    compute_timer = _resolve_section_timer(device, should_use_cuda_events)
    readout_timer = _resolve_section_timer(device, should_use_cuda_events)
    previous_training_mode = model.training
    if mode == 'inference':
        model.eval()
    else:
        model.train()
    try:
        total_samples = _measure_samples(
            lambda: _run_readouts(model.forward(x), normalized_readout_names),
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            trials=trials,
            mode=mode,
            device=device,
            synchronize=should_synchronize,
            timer='wall_clock',
        )

        encode_samples = _measure_samples(
            lambda: model.curry_forward(x),
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            trials=trials,
            mode=mode,
            device=device,
            synchronize=should_synchronize,
            timer='wall_clock',
        )

        curried_forward = model.curry_forward(x)
        compute_samples = _measure_samples(
            curried_forward,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            trials=trials,
            mode=mode,
            device=device,
            synchronize=should_synchronize,
            timer=compute_timer,
        )

        readout_samples = _measure_prepared_samples(
            curried_forward,
            lambda forward_state: _run_readouts(forward_state, normalized_readout_names),
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            trials=trials,
            mode=mode,
            device=device,
            synchronize=should_synchronize,
            timer=readout_timer,
        )
    finally:
        model.train(previous_training_mode)

    return ForwardReadoutLatency(
        mode=mode,
        device_type=device.type,
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        readout_names=normalized_readout_names,
        warmup_iterations=warmup_iterations,
        trials=trials,
        timers={
            'total': 'wall_clock',
            'encode': 'wall_clock',
            'compute': compute_timer,
            'readout': readout_timer,
        },
        total=LatencyStats.from_samples(total_samples),
        encode=LatencyStats.from_samples(encode_samples),
        compute=LatencyStats.from_samples(compute_samples),
        readout=LatencyStats.from_samples(readout_samples),
    )
