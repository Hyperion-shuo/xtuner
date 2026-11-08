"""Migration contracts independent of CRG, verifier policy and GPU availability."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock
import pytest
from xtuner.v1.rl.agent_loop_manager.turn_async import TurnManager, TurnManagerConfig, ManagerThread
from xtuner.v1.rl.agent_loop_manager.turn_async_config import TurnAsyncOptions
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManagerConfig
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.turn_async_loop import validate_segment
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.train.turn_async_trainer import optimizer_update_succeeded, TurnRLColocateTrainer


def test_new_mode_rejects_multiple_runners_without_touching_old_mode():
    # Use the validator directly: no tokenizer/model/runner resources are built.
    cfg = TurnManagerConfig.model_construct(tasks=[NS(), NS()])
    with pytest.raises(ValueError, match='exactly one runner'):
        cfg.single_runner()
    assert len(AgentLoopManagerConfig.model_construct(tasks=[NS(), NS()]).tasks) == 2
    assert RolloutConfig.model_fields['turn_level_async'].default is False


def test_dense_trace_keeps_history_and_routing_is_an_explicit_capability():
    sample = NS(input_ids=[7,8,9], labels=[-100,8,9], response_ids=[8,9], logprobs=[0,-.2,-.7],
        response_model_steps=[0,4], routed_experts=None,
        extra_fields={'response_policy_versions':[0,1], 'routing_source_request_id':'r'})
    assert validate_segment(sample)['policy_versions'] == [0,1]
    assert sample.logprobs == [0,-.2,-.7] and sample.response_model_steps == [0,4]
    with pytest.raises(ValueError, match='routing'):
        validate_segment(sample, require_routing=True)
    sample.response_ids = [7,8]
    with pytest.raises(ValueError, match='positions'):
        validate_segment(sample)


def test_new_weights_publish_only_after_sync_and_skip_keeps_version():
    async def run():
        obj = TurnManager.__new__(TurnManager)
        obj.version, obj.source_optimizer_step, obj.epoch = 0,0,0
        obj.last_successful_optimizer_step = obj.synced_optimizer_step = None
        obj.event = Mock()
        obj._drop_expired_pending = AsyncMock()
        obj.task_runners = [NS(agent_loop=NS(resume_new_trials=Mock()))]
        control = AsyncMock()
        obj._rollout_controller = NS(turn_control=NS(remote=control), continue_generation=NS(remote=AsyncMock()))
        await obj.record_optimizer_outcome(1, True)
        await obj._publish(0)  # optimizer succeeded but sync interval has not elapsed
        assert obj.version == 0 and obj.source_optimizer_step == 0
        await obj.record_weights_synced()
        await obj._publish(1)
        assert obj.version == 1 and obj.source_optimizer_step == 1
        await obj.record_optimizer_outcome(2, False)
        await obj.record_weights_synced()
        await obj._publish(2)
        assert obj.version == 1 and obj.source_optimizer_step == 1
        with pytest.raises(RuntimeError, match='unknown'):
            await obj.record_optimizer_outcome(3, None)
    asyncio.run(run())


def test_optimizer_evidence_cannot_invent_a_successful_update():
    logs = lambda *norms: [{'train_metrics':[{'grad_norm':n}]} for n in norms]
    assert optimizer_update_succeeded(logs(.3,.3), 1) is True
    assert optimizer_update_succeeded(logs(2,2), 1) is False
    assert optimizer_update_succeeded(logs(float('nan')), None) is False
    assert optimizer_update_succeeded(logs(.3,2), 1) is None
    assert optimizer_update_succeeded([], 1) is None


def test_consumption_releases_only_owned_batch_sessions():
    trainer = TurnRLColocateTrainer.__new__(TurnRLColocateTrainer)
    trainer._release_trace_sessions = Mock()
    trainer._release_all_trace_sessions = Mock()
    trainer._release_trace_sessions_after_train_batch([[NS(session_id=1), NS(session_id=2)]])
    trainer._release_trace_sessions.assert_called_once_with(['1','2'])
    trainer._release_all_trace_sessions.assert_not_called()


def test_failed_manager_factory_does_not_leak_thread():
    import threading
    before = {t.ident for t in threading.enumerate()}
    def bad(): raise ValueError('bad config')
    with pytest.raises(ValueError, match='bad config'):
        ManagerThread(bad)
    assert not [t for t in threading.enumerate() if t.ident not in before and t.name == 'turn-rollout']


def test_partial_open_failure_rolls_back_slow_and_fast_entrypoints():
    from xtuner.v1.rl.rollout.controller import RolloutController
    async def run():
        actions = []
        async def remote(name, action, **kwargs):
            actions.append((name, action))
            if action == 'status': return {'instance_id':name}
            if action == 'open_admission':
                if name == 'failed': raise RuntimeError('failed entrypoint')
                await asyncio.sleep(.02)
                actions.append((name, 'opened'))
            return {'instance_id':name}
        workers = [NS(actor=NS(turn_control=NS(remote=lambda action, _name=name, **kw: remote(_name,action,**kw))))
                   for name in ('failed','slow')]
        ctl = RolloutController.__new__(RolloutController)
        ctl.registry = NS(active_workers=lambda:workers, all_workers=lambda:workers)
        ctl.health_manager = NS(pause=Mock(), resume=Mock())
        ctl._turn_instances = None
        with pytest.raises(RuntimeError, match='failed entrypoint'):
            await ctl.turn_control('open_admission',version=1, source_optimizer_step=1,epoch=1)
        assert ('slow','opened') in actions
        assert all((name,'close_admission') in actions for name in ('failed','slow'))
        assert actions.index(('slow','opened')) < actions.index(('slow','close_admission'))
        ctl.health_manager.resume.assert_not_called()
    asyncio.run(run())


def test_dispatch_uses_publication_counter_after_optimizer_skip():
    from unittest.mock import patch
    async def run():
        obj = TurnManager.__new__(TurnManager)
        obj.options = TurnAsyncOptions(batch_wait=AsyncMock(return_value=False))
        obj.event = obj.disposition = Mock()
        obj.version, obj.source_optimizer_step = 2, 4
        obj._publish, obj._collect = AsyncMock(), AsyncMock()
        obj._drain = AsyncMock(return_value=0)
        obj.sample_pool_groups = 0
        obj.consumed, obj.raw, obj.pending = [], [], {}
        sample = NS(group_id=1, session_id=1, input_ids=[0,1], response_ids=[1],
            response_model_steps=[4],reward={'score':1},
            extra_fields={'trial':{'task_id':'t','data_source':'test'},
                'response_policy_versions':[2], 'routing_source_version':2,'routing_source_request_id':'r'})
        obj.ready = [[sample]]
        loop = NS(set_generation_context=Mock())
        obj.task_runners = [NS(agent_loop=loop)]
        store = NS(turn_memory=NS(remote=AsyncMock(return_value={})))
        with patch('xtuner.v1.rl.agent_loop_manager.turn_async.get_store', return_value=store):
            await obj.produce_batch(1,5,model_step=4)
        obj.options.batch_wait.assert_awaited_once_with(obj,5)
        loop.set_generation_context.assert_called_once_with(5,2)
        assert sample.response_model_steps == [4]  # preserve historical optimizer provenance
    asyncio.run(run())


def test_remote_trial_loops_are_rejected_before_build():
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.turn_async_loop import TurnAgentInSandboxLoopConfig
    loop = TurnAgentInSandboxLoopConfig.model_construct(hf_checkpoint='unused', cpu_resources=object())
    cfg = TurnManagerConfig.model_construct(tasks=NS(agent_loop_config=loop))
    with pytest.raises(ValueError, match='local'):
        cfg.single_runner()


def test_adapter_failure_cancels_and_joins_other_trials():
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.turn_async_loop import TurnAgentInSandboxLoop
    from xtuner.v1.rl.agent_loop_manager.turn_async_config import TurnTrialOptions
    async def run():
        loop = TurnAgentInSandboxLoop.__new__(TurnAgentInSandboxLoop)
        loop.running, loop.sampling_window, loop.policy_version = {},0,0
        loop._sample_semaphore = asyncio.Semaphore(2)
        loop.trial_admission = asyncio.Event()
        loop.trial_admission.set()
        loop.event = loop.disposition = Mock()
        entered, cleaned = asyncio.Event(), []
        async def before(state,item,identity):
            if state.rollout_id == 0:
                await entered.wait()
                raise RuntimeError('task adapter failed')
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.01)
                cleaned.append(state.rollout_id)
        loop.options = TurnTrialOptions(run_id='test', before_trial=before)
        states=[NS(rollout_id=i,group_id=1,extra_fields={'rollout_item':NS(id='t',data_source='test',metadata={})})
                for i in range(2)]
        with pytest.raises(RuntimeError, match='task adapter failed'):
            await loop.generate_group(states)
        assert cleaned == [1] and not loop.running
    asyncio.run(run())
