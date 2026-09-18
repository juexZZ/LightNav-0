"""Per-episode state and differentiable writer contracts for future memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

MemoryState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class MemoryObservation:
    frame_ids: tuple[int, int]
    timestamp_s: float
    features: torch.Tensor


@dataclass(frozen=True)
class MemoryReadout:
    tokens: torch.Tensor
    state_tokens: int
    cache_elements: int = 0


@dataclass(frozen=True)
class MemoryContext:
    history: MemoryReadout
    current: MemoryObservation
    episode_id: str
    unique_writes: int


class MemoryWriter(nn.Module, ABC):
    """Functional updates must not mutate input state, including on budget failure.

    State contains named tensors only. A concrete adapter can flatten its layerwise
    KV/state into this mapping. Readout tokens must already use the LLM hidden width.
    No learned writer, checkpoint, video layout or DeepStack policy is assumed here.
    """

    @abstractmethod
    def initial_state(self, reference: torch.Tensor) -> MemoryState:
        raise NotImplementedError

    @abstractmethod
    def update(self, state: Mapping[str, torch.Tensor], observation: MemoryObservation) -> MemoryState:
        raise NotImplementedError

    @abstractmethod
    def read(self, state: Mapping[str, torch.Tensor]) -> MemoryReadout:
        raise NotImplementedError


class MemorySession:
    """Owns one episode's state; writer parameters may be shared across sessions."""

    def __init__(self, writer: MemoryWriter, *, writer_id: str, hidden_size: int,
                 max_state_tokens: int, max_read_tokens: int, episode_id: str):
        if not writer_id or min(hidden_size, max_state_tokens, max_read_tokens) < 1:
            raise ValueError('Writer identity and positive dimensions/budgets are required')
        self.writer = writer
        self.writer_id = writer_id
        self.hidden_size = hidden_size
        self.max_state_tokens = max_state_tokens
        self.max_read_tokens = max_read_tokens
        self.reset(episode_id)

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError('An explicit episode ID is required')
        self.episode_id = episode_id
        self.state: MemoryState | None = None
        self.seen: dict[tuple[int, int], int] = {}
        self.last_frame = -1
        self.last_timestamp = float('-inf')

    def _validate_observation(self, observation: MemoryObservation) -> None:
        frames, features = observation.frame_ids, observation.features
        if not isinstance(frames, tuple) or len(frames) != 2 or not all(isinstance(frame, int) for frame in frames) or not 0 <= frames[0] <= frames[1]:
            raise ValueError('frame_ids must be a causal pair of non-negative absolute frame IDs')
        if not torch.is_tensor(features) or features.ndim != 2 or features.shape[1] != self.hidden_size or not torch.is_floating_point(features):
            raise ValueError('Observation features must be floating [tokens, hidden_size] tensors')
        if features.shape[0] == 0 or not 0 <= observation.timestamp_s < float('inf'):
            raise ValueError('An observation needs nonempty features and a finite absolute timestamp')

    def _read_checked(self, state: MemoryState) -> MemoryReadout:
        if not isinstance(state, dict) or any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
            raise TypeError('Memory state must be a dictionary of named tensors')
        output = self.writer.read(state)
        if not torch.is_tensor(output.tokens) or output.tokens.ndim != 2 or output.tokens.shape[1] != self.hidden_size or not torch.is_floating_point(output.tokens):
            raise ValueError('Memory readout must be [tokens, hidden_size]')
        if not isinstance(output.state_tokens, int) or not isinstance(output.cache_elements, int) or output.state_tokens < output.tokens.shape[0] or output.cache_elements < 0:
            raise ValueError('Invalid memory state/cache accounting')
        if output.state_tokens > self.max_state_tokens or output.tokens.shape[0] > self.max_read_tokens:
            raise OverflowError('Memory budget exceeded; never silently truncate history or current vision')
        return output

    def append_history(self, observation: MemoryObservation) -> bool:
        self._validate_observation(observation)
        if observation.frame_ids in self.seen:
            if self.seen[observation.frame_ids] != observation.features.shape[0]:
                raise ValueError('A repeated tubelet changed its feature layout')
            return False
        if observation.frame_ids[-1] <= self.last_frame or observation.timestamp_s < self.last_timestamp:
            raise ValueError('New history must advance causally; out-of-order replay is forbidden')
        previous = self.state if self.state is not None else self.writer.initial_state(observation.features)
        candidate = self.writer.update(previous, observation)
        self._read_checked(candidate)
        self.state = candidate
        self.seen[observation.frame_ids] = observation.features.shape[0]
        self.last_frame = observation.frame_ids[-1]
        self.last_timestamp = observation.timestamp_s
        return True

    def context(self, current: MemoryObservation) -> MemoryContext:
        self._validate_observation(current)
        if self.last_frame >= current.frame_ids[-1] or self.last_timestamp > current.timestamp_s:
            raise ValueError('Memory must contain only history older than the current observation')
        state = self.state if self.state is not None else self.writer.initial_state(current.features)
        readout = self._read_checked(state)
        if readout.tokens.device != current.features.device or readout.tokens.dtype != current.features.dtype:
            raise ValueError('History and current features must have matching device and dtype')
        return MemoryContext(readout, current, self.episode_id, len(self.seen))

    def detach_state(self) -> None:
        if self.state is not None:
            self.state = {name: value.detach() for name, value in self.state.items()}

    def snapshot(self) -> dict:
        return {
            'format_version': 1, 'writer_id': self.writer_id, 'hidden_size': self.hidden_size,
            'episode_id': self.episode_id, 'last_frame': self.last_frame,
            'last_timestamp': self.last_timestamp,
            'seen': [{'frames': list(frames), 'tokens': count} for frames, count in self.seen.items()],
            'state': None if self.state is None else {
                name: value.detach().cpu().clone() for name, value in self.state.items()
            },
        }

    def restore(self, snapshot: dict, *, device: torch.device | str = 'cpu') -> None:
        expected = (1, self.writer_id, self.hidden_size, self.episode_id)
        actual = tuple(snapshot.get(key) for key in ('format_version', 'writer_id', 'hidden_size', 'episode_id'))
        if actual != expected:
            raise ValueError('Snapshot format/writer/hidden size/episode identity mismatch')
        rows = snapshot['seen']
        seen = {tuple(row['frames']): row['tokens'] for row in rows}
        if len(seen) != len(rows) or any(len(frames) != 2 or not all(isinstance(frame, int) for frame in frames) or not 0 <= frames[0] <= frames[1] or not isinstance(count, int) or count < 1 for frames, count in seen.items()):
            raise ValueError('Invalid snapshot tubelet index')
        last_frame = max((frames[-1] for frames in seen), default=-1)
        if last_frame != snapshot['last_frame'] or bool(seen) != (snapshot['state'] is not None):
            raise ValueError('Snapshot history metadata disagrees with its state')
        timestamp = snapshot['last_timestamp']
        if not isinstance(timestamp, (int, float)) or (seen and not 0 <= timestamp < float('inf')) or (not seen and timestamp != float('-inf')):
            raise ValueError('Invalid snapshot timestamp')
        candidate = None if snapshot['state'] is None else {
            name: value.detach().to(device).clone() for name, value in snapshot['state'].items()
        }
        if candidate is not None:
            self._read_checked(candidate)
        self.state, self.seen = candidate, seen
        self.last_frame, self.last_timestamp = last_frame, snapshot['last_timestamp']
