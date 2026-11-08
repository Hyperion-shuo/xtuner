import asyncio
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from xtuner.v1.rl.rollout.trace_shards import ShardedTraceStore, shard_count


def test_stable_direct_session_routing_and_reservation():
    actors=[Mock() for _ in range(4)]
    store=ShardedTraceStore(actors)
    for sid in ['s','long-session','1234']:
        index=store.index(sid)
        assert index==ShardedTraceStore(actors).index(sid)
        store.prepare_turn.remote(sid,'P','r')
        actors[index].prepare_turn.remote.assert_called_with(sid,'P','r')
        store.reserve_turn.remote('r',10,session_id=sid)
        actors[index].reserve_turn.remote.assert_called_with('r',10,session_id=sid)
        store.release_turn_reservation.remote('r',session_id=sid)
        actors[index].release_turn_reservation.remote.assert_called_with('r',session_id=sid)
    with pytest.raises(ValueError,match='session_id'):store.reserve_turn.remote('r',10)


def test_invalid_shard_count(monkeypatch):
    for n in ('0','65','-1'):
        monkeypatch.setenv('XTUNER_TRACE_SHARDS',n)
        with pytest.raises(ValueError):shard_count()


def test_grouped_cleanup_and_aggregation_return_ray_refs(monkeypatch):
    import xtuner.v1.rl.rollout.trace_shards as module
    actors=[Mock() for _ in range(4)];store=ShardedTraceStore(actors)
    aggregate=Mock(return_value='aggregate-ref')
    monkeypatch.setattr(module,'_collect',SimpleNamespace(remote=aggregate))
    ids=['a','b','a','c']
    assert store.begin_release_sessions.remote(ids)=='aggregate-ref'
    assert aggregate.call_args.args[0]=='begin_release_sessions'
    seen=[]
    for actor in actors:
        if actor.begin_release_sessions.remote.called:
            seen.extend(actor.begin_release_sessions.remote.call_args.args[0])
    assert set(seen)=={'a','b','c'} and len(seen)==3
    assert store.turn_memory.remote()=='aggregate-ref'
    assert len(aggregate.call_args.args[1:])==4


@pytest.mark.parametrize('count', [1, 4])
def test_acquire_preserves_total_budget_and_rejects_mismatched_topology(monkeypatch, count):
    import xtuner.v1.rl.rollout.trace_shards as module
    monkeypatch.setenv('XTUNER_TRACE_SHARDS',str(count))
    monkeypatch.setenv('XTUNER_TRACE_HARD_BYTES','103')
    registry={}
    quotas=[]
    class Actor:
        def options(self, **kwargs):
            self.name=kwargs['name']
            return self
        def remote(self, byte_limit, topology):
            quotas.append(byte_limit)
            actor=SimpleNamespace(trace_topology=SimpleNamespace(remote=lambda:topology))
            registry[self.name]=actor
            return actor
    def get(name, **kwargs):
        if name not in registry: raise ValueError('not found')
        return registry[name]
    monkeypatch.setattr(module.ray,'get_actor',get)
    monkeypatch.setattr(module.ray,'get',lambda refs:refs)
    store=module.acquire_shards(Actor(),'test','test',create=True)
    assert sum(quotas)==103 and len(quotas)==count
    assert max(quotas)-min(quotas)<=1
    assert isinstance(store,ShardedTraceStore)==(count>1)
    monkeypatch.setenv('XTUNER_TRACE_HARD_BYTES','104')
    with pytest.raises(RuntimeError,match='topology or budget'):
        module.acquire_shards(Actor(),'test','test',create=False)


@pytest.mark.parametrize('method,expected', [('oldest_session_versions',{}),
    ('begin_release_sessions',[]),('continue_release_sessions',{'pending':[],'processed':0}),('release_sessions',[])])
def test_empty_group_aggregation(method,expected):
    import xtuner.v1.rl.rollout.trace_shards as module
    assert module._collect._function(method)==expected


def test_store_handle_cache_rejects_topology_changes(monkeypatch):
    import xtuner.v1.rl.rollout.trace_store as module
    monkeypatch.setenv('XTUNER_TRACE_SHARDS','1')
    monkeypatch.setenv('XTUNER_TRACE_HARD_BYTES','128')
    monkeypatch.setattr(module,'_handle_cache',object())
    monkeypatch.setattr(module,'_handle_topology',(1,128),raising=False)
    assert module.get_store() is module._handle_cache
    monkeypatch.setenv('XTUNER_TRACE_HARD_BYTES','129')
    with pytest.raises(RuntimeError,match='changed within'):
        module.get_store()
