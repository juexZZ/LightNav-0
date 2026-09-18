from __future__ import annotations

import pytest
import torch
from torch import nn

from lightnav.memory import MemoryObservation, MemoryReadout, MemorySession, MemoryWriter


class ToyWriter(MemoryWriter):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def initial_state(self, reference):
        return {'tokens': reference.new_empty((0, reference.shape[1]))}

    def update(self, state, observation):
        return {'tokens': torch.cat([state['tokens'], observation.features * self.scale])}

    def read(self, state):
        return MemoryReadout(state['tokens'], state['tokens'].shape[0])


def session(writer=None, episode_id='episode', budget=20):
    return MemorySession(writer or ToyWriter(), writer_id='toy-v1', hidden_size=4,
                         max_state_tokens=budget, max_read_tokens=budget, episode_id=episode_id)


def observation(end, tokens=2):
    return MemoryObservation((max(0, end - 1), end), float(end), torch.ones(tokens, 4))


def test_empty_history_keeps_current_separate():
    memory = session()
    current = observation(1)
    context = memory.context(current)
    assert context.history.tokens.shape == (0, 4)
    assert context.current is current
    assert context.unique_writes == 0


def test_incremental_unique_writes_and_causality():
    memory = session()
    assert memory.append_history(observation(1))
    assert not memory.append_history(observation(1))
    assert memory.append_history(observation(3, tokens=3))
    assert memory.context(observation(5)).history.tokens.shape == (5, 4)
    with pytest.raises(ValueError, match='out-of-order'):
        memory.append_history(observation(2))
    with pytest.raises(ValueError, match='older'):
        memory.context(observation(3))
    with pytest.raises(ValueError, match='layout'):
        memory.append_history(observation(1, tokens=3))


def test_budget_failure_is_transactional():
    memory = session(budget=3)
    memory.append_history(observation(1))
    with pytest.raises(OverflowError):
        memory.append_history(observation(3))
    assert memory.last_frame == 1
    assert len(memory.seen) == 1
    assert memory.context(observation(5)).history.state_tokens == 2


def test_frozen_backbone_still_passes_gradients_to_writer():
    memory = session()
    backbone = nn.Linear(4, 1, bias=False).requires_grad_(False)
    with torch.no_grad():
        backbone.weight.fill_(1.0)
    before = memory.writer.scale.detach().clone()
    optimizer = torch.optim.SGD(memory.writer.parameters(), lr=0.1)
    memory.append_history(observation(1))
    loss = backbone(memory.context(observation(3)).history.tokens).sum()
    loss.backward()
    assert memory.writer.scale.grad.abs().item() > 0
    assert backbone.weight.grad is None
    optimizer.step()
    assert not torch.equal(before, memory.writer.scale.detach())


def test_sessions_do_not_share_state_and_reset_clears_it():
    writer = ToyWriter()
    first, second = session(writer, 'first'), session(writer, 'second')
    first.append_history(observation(1))
    assert second.context(observation(3)).history.state_tokens == 0
    first.reset('next')
    assert first.context(observation(1)).history.state_tokens == 0


def test_snapshot_roundtrip_identity_and_detach():
    memory = session()
    memory.append_history(observation(1))
    snapshot = memory.snapshot()
    assert snapshot['state']['tokens'].device.type == 'cpu'
    assert not snapshot['state']['tokens'].requires_grad
    restored = session()
    restored.restore(snapshot)
    assert torch.equal(restored.context(observation(3)).history.tokens,
                       memory.context(observation(3)).history.tokens)
    assert not restored.append_history(observation(1))
    with pytest.raises(ValueError, match='identity'):
        session(episode_id='other').restore(snapshot)
    memory.detach_state()
    assert not memory.state['tokens'].requires_grad


def test_shape_and_future_timestamp_checks():
    memory = session()
    with pytest.raises(ValueError, match='hidden_size'):
        memory.append_history(MemoryObservation((0, 1), 1.0, torch.ones(2, 5)))
    with pytest.raises(ValueError, match='finite'):
        memory.append_history(MemoryObservation((0, 1), float('nan'), torch.ones(2, 4)))
    memory.append_history(MemoryObservation((0, 1), 10.0, torch.ones(2, 4)))
    with pytest.raises(ValueError, match='older'):
        memory.context(observation(3))


def test_restore_rejects_nonfinite_timestamp_and_budget_overflow():
    memory = session()
    memory.append_history(observation(1))
    snapshot = memory.snapshot()
    with pytest.raises(OverflowError):
        session(budget=1).restore(snapshot)
    snapshot['last_timestamp'] = float('nan')
    with pytest.raises(ValueError, match='timestamp'):
        session().restore(snapshot)
