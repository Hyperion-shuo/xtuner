"""Persistent single-runner scheduling across colocated GPU transitions.

Extracted from the verified CRG turn scheduler; task policy is injected.
"""
from __future__ import annotations
import asyncio
import inspect
import math
import threading
import time
from pathlib import Path
import json
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager, AgentLoopManagerConfig
from xtuner.v1.rl.agent_loop_manager.produce_utils import ProduceBatchResult, manager_state_path, task_checkpoint_path
from xtuner.v1.rl.rollout.trace_store import release_existing_sessions, get_store
from pydantic import Field, model_validator
from .turn_async_config import TurnAsyncOptions
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.turn_async_loop import TurnAgentInSandboxLoop, TurnAgentInSandboxLoopConfig, validate_segment

class TurnManager(AgentLoopManager):
    def __init__(self, *, options=None, **kwargs):
        super().__init__(**kwargs)
        self.options = options or TurnAsyncOptions()
        self.event = self.options.event
        self.disposition = self.options.disposition
        if len(self.task_runners) != 1:
            raise ValueError("turn-level colocated mode supports exactly one runner")
        self.pending = {}
        self.ready = []
        self.raw = []
        self.version = -1
        self.source_optimizer_step = -1
        self.epoch = 0
        self.launched = 0
        self.completed = 0
        if not isinstance(self.task_runners[0].agent_loop, TurnAgentInSandboxLoop):
            raise TypeError("turn-level manager requires a local TurnAgentInSandboxLoop")
        self.max_groups = math.inf if self.options.max_groups is None else self.options.max_groups
        self.sample_pool_groups = self.options.pool_groups
        self.deadline = ((self.options.started_at or time.time()) + self.options.budget_seconds
                         if self.options.budget_seconds else math.inf)
        self.consumed = []
        self.closing = False
        self.raw_ready_since = None
        self.create_backoff_until = 0.0
        self.last_successful_optimizer_step = None
        self.synced_optimizer_step = None

    async def _release(self, group):
        session_ids = list(dict.fromkeys(str(s.session_id) for s in group if s.session_id is not None))
        if session_ids:
            await self._rollout_controller.retire_route_sessions.remote(session_ids)
            started = time.monotonic()
            released = await release_existing_sessions(session_ids)
            self.event("trace_cleanup", requested=len(session_ids), released=len(released),
                       duration=time.monotonic() - started)
            if self.options.trace_metrics:
                self.event("trace_store_metrics", **(await get_store().trace_metrics.remote()))

    def _valid(self, group):
        valid = []
        for sample in group:
            if sample.status != Status.COMPLETED or not sample.extra_fields.get("cleanup_ok"):
                continue
            try:
                validate_segment(sample, require_routing=self.options.require_routing)
            except ValueError:
                continue
            valid.append(sample)
        return valid

    def _expired(self, group):
        return any(self.version - min(s.extra_fields["response_policy_versions"]) > self.options.max_staleness
                   for s in group)

    def _score(self, group):
        return self.options.group_filter(group)

    async def _collect(self, batch_size):
        for future in list(self.pending):
            if not future.done():
                continue
            original = self.pending[future]
            result = await future
            # Every Trial in this future is terminal. Keep trace data for training,
            # but free its router binding and reject any late HTTP retries.
            await self._rollout_controller.retire_route_sessions.remote(
                [str(s.session_id) for s in original if s.session_id is not None])
            delay = self.options.result_backoff(result) if self.options.result_backoff else 0
            if delay:
                self.create_backoff_until = time.monotonic() + delay
                self.event("sandbox_create_backoff", seconds=delay, group_id=original[0].group_id)
            self.completed += 1
            valid = self._valid(result)
            keep_sessions = {s.session_id for s in valid}
            await self._release([s for s in original if s.session_id not in keep_sessions])
            rejected = [s for s in result if not any(s is v for v in valid)]
            if rejected:
                self.disposition([rejected], filtered=True, filter_reason="invalid_trace_or_terminal")
            if not valid or self._expired(valid):
                if valid:
                    self.disposition([valid], filtered=True, filter_reason="group_expired")
                await self._release(valid)
                self.event("group_discard", group_id=original[0].group_id, reason="failed_or_expired")
            elif self._score(valid):
                self.ready.append(valid)
            else:
                self.disposition([valid], filtered=True, filter_reason="group_reward_filter")
                await self._release(valid)
            del self.pending[future]

    async def _publish(self, source_optimizer_step):
        if self.synced_optimizer_step is not None:
            source_optimizer_step = self.synced_optimizer_step
        version = self.version + int(source_optimizer_step != self.source_optimizer_step)
        ctl = self._rollout_controller
        await self._drop_expired_pending(version)
        await ctl.continue_generation.remote()
        await ctl.turn_control.remote("prepare_version", version=version,
                                      source_optimizer_step=source_optimizer_step, epoch=self.epoch)
        await ctl.turn_control.remote("open_admission", version=version,
                                      source_optimizer_step=source_optimizer_step, epoch=self.epoch)
        self.version = version
        self.source_optimizer_step = source_optimizer_step
        self.event("version_published", version=version, source_optimizer_step=source_optimizer_step, epoch=self.epoch)
        self.task_runners[0].agent_loop.resume_new_trials()

    async def record_optimizer_outcome(self, train_step, updated):
        if updated is None:
            raise RuntimeError("cannot publish unknown optimizer outcome")
        if updated:
            self.last_successful_optimizer_step = train_step
        elif self.last_successful_optimizer_step is None:
            self.last_successful_optimizer_step = max(self.source_optimizer_step, 0)
        self.event("optimizer_version_source", train_step=train_step, updated=updated,
              source_optimizer_step=self.last_successful_optimizer_step)

    async def record_weights_synced(self):
        if self.last_successful_optimizer_step is None:
            raise RuntimeError("weight sync requires a known optimizer outcome")
        self.synced_optimizer_step = self.last_successful_optimizer_step

    async def _drop_expired_pending(self, next_version):
        if not self.pending:
            return
        loop = self.task_runners[0].agent_loop
        sessions = [str(s.session_id) for group in self.pending.values() for s in group
                    if s.session_id is not None
                    and (str(s.session_id) in loop.running or s.status == Status.COMPLETED)]
        oldest = await get_store().oldest_session_versions.remote(sessions)
        for future, group in list(self.pending.items()):
            if not any(next_version - oldest[str(s.session_id)] > self.options.max_staleness for s in group
                       if str(s.session_id) in oldest):
                continue
            session_ids = {str(s.session_id) for s in group if s.session_id is not None}
            await asyncio.gather(*[
                self._rollout_controller.turn_control.remote("cancel_session", session_id=s, reason="group_expired")
                for s in session_ids])
            future.cancel()
            await asyncio.gather(future, return_exceptions=True)
            await self._release(group)
            del self.pending[future]
            self.event("group_discard", group_id=group[0].group_id, reason="live_group_expired",
                  next_version=next_version)

    async def _drain(self):
        self.epoch += 1
        self.task_runners[0].agent_loop.pause_new_trials()
        ctl = self._rollout_controller
        await ctl.turn_control.remote("close_admission", epoch=self.epoch)
        start = time.monotonic()
        deadline = start + self.options.drain_seconds
        while True:
            statuses = await ctl.turn_control.remote("status")
            if all(s["active"] == 0 for s in statuses):
                break
            if time.monotonic() >= deadline:
                sessions = {t["session_id"] for s in statuses for t in s["requests"]}
                await asyncio.gather(*[
                    ctl.turn_control.remote("cancel_session", session_id=session, reason="drain_timeout")
                    for session in sessions
                ])
                for session in sessions:
                    task = self.task_runners[0].agent_loop.running.get(session)
                    if task is not None:
                        task.cancel()
                    self.event("drain_cancel", session_id=session)
                break
            await asyncio.sleep(0.2)
        await ctl.turn_control.remote("drain", timeout=60.0)
        duration = time.monotonic() - start
        self.event("drained", epoch=self.epoch, duration=duration)
        return duration

    async def produce_batch(self, batch_size, train_step, *, model_step):
        started = time.monotonic()
        pool_capacity = self.sample_pool_groups or 2 * batch_size
        if pool_capacity < batch_size:
            raise ValueError("sample pool must hold at least one training batch")
        await self._publish(model_step)
        loop = self.task_runners[0].agent_loop
        loop.set_generation_context(train_step, self.version)
        for queue in (self.ready, self.raw):
            for group in list(queue):
                if self._expired(group):
                    queue.remove(group)
                    self.disposition([group], filtered=True, filter_reason="ready_group_expired")
                    await self._release(group)
        async def should_wait():
            if self.options.batch_wait is None:
                return False
            result = self.options.batch_wait(self, train_step)
            return await result if inspect.isawaitable(result) else result

        while len(self.ready) < batch_size or await should_wait():
            if time.time() >= self.deadline:
                raise TimeoutError("turn-level sampling wall-clock budget reached")
            await self._collect(batch_size)
            memory = await get_store().turn_memory.remote()
            can_start_group = (time.monotonic() >= self.create_backoff_until
                               and memory["routing_bytes"] < self.options.trace_soft_bytes)
            # Deferred unscored groups also count toward the admission budget.
            while (len(self.pending) + len(self.ready) + len(self.raw) < pool_capacity
                   and self.launched < self.max_groups and can_start_group):
                states = await self.task_runners[0].sampler.sample(self.task_runners[0].task_name)
                self.launched += 1
                future = asyncio.create_task(loop.generate_group(states))
                self.pending[future] = states
                self.event("group_started", group_id=states[0].group_id, launched=self.launched,
                      task_id=states[0].extra_fields["rollout_item"].id,
                      data_source=states[0].extra_fields["rollout_item"].data_source)
            if not self.pending and len(self.ready) < batch_size and self.launched >= self.max_groups:
                raise RuntimeError("No complete trainable batch within the sampling group budget")
            await asyncio.sleep(0.1)
        pause_time = await self._drain()
        await self._collect(batch_size)
        if self.options.batch_order:
            self.options.batch_order(self, train_step)
        batch, self.ready = self.ready[:batch_size], self.ready[batch_size:]
        ids = [group[0].group_id for group in batch]
        if len(set(ids)) != len(ids) or any(key in self.consumed for key in ids):
            raise RuntimeError("A prompt group was selected more than once")
        self.consumed.extend(ids)
        self.disposition(batch, selected_for_training=True, training_step=train_step)
        for group in batch:
            for sample in group:
                self.event("trace_selected", train_step=train_step, group_id=sample.group_id,
                      session_id=str(sample.session_id),
                      data_source=sample.extra_fields["trial"]["data_source"],
                      task_id=sample.extra_fields["trial"]["task_id"],
                      generation_versions=sorted(set(sample.extra_fields["response_policy_versions"])),
                      source_optimizer_steps=sorted(set(sample.response_model_steps)),
                      routing_source_version=sample.extra_fields["routing_source_version"],
                      routing_source_request_id=sample.extra_fields["routing_source_request_id"],
                      token_count=len(sample.input_ids), loss_tokens=len(sample.response_ids),
                      raw_reward=sample.extra_fields.get("raw_reward"), reward=sample.reward,
                      reward_source=sample.extra_fields.get("reward_source"))
        self.event("batch_selected", train_step=train_step, group_ids=[g[0].group_id for g in batch],
              elapsed=time.monotonic() - started,
              trace_memory=await get_store().turn_memory.remote())
        return ProduceBatchResult(rollout_states=batch, group_gen_pause_time_s=pause_time)

    async def save(self, checkpoint_path, model_step, *, no_save_replay_buffer=False):
        if not no_save_replay_buffer:
            raise ValueError("turn-level live agents are not checkpoint-serializable")
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)
        for task in self.task_runners:
            target = task_checkpoint_path(path, task.task_name)
            target.mkdir(parents=True, exist_ok=True)
            task.sampler.save(target)
        manager_state_path(path).write_text(json.dumps({"model_step": model_step, "replay_buffer_saved": False,
                                                       "launched": self.launched, "consumed": self.consumed,
                                                       "pending_sessions": list(self.task_runners[0].agent_loop.running),
                                                       "version": self.version,
                                                       "source_optimizer_step": self.source_optimizer_step,
                                                       "last_successful_optimizer_step": self.last_successful_optimizer_step,
                                                       "epoch": self.epoch}))

    async def resume(self, checkpoint_path):
        step = await super().resume(checkpoint_path)
        state = json.loads(manager_state_path(Path(checkpoint_path)).read_text())
        self.launched = state["launched"]
        self.consumed = state["consumed"]
        self.version = state.get("version", -1)
        self.source_optimizer_step = state.get("source_optimizer_step", -1)
        self.last_successful_optimizer_step = state.get("last_successful_optimizer_step")
        # Resume loads training checkpoint weights before the first publication.
        self.synced_optimizer_step = self.last_successful_optimizer_step
        # New server actors start at epoch zero; Python tasks are not restored.
        self.epoch = 0
        self.event("checkpoint_resumed", model_step=step, live_agents_restored=False)
        return step

    async def close(self):
        if self.closing:
            return
        self.closing = True
        self.task_runners[0].agent_loop.pause_new_trials()
        errors = []
        async def attempt(operation):
            try:
                return await operation
            except Exception as exc:
                errors.append(exc)

        await attempt(self._rollout_controller.turn_control.remote("close_admission", epoch=self.epoch + 1))
        loop = self.task_runners[0].agent_loop
        cancelled = await asyncio.gather(*[
            self._rollout_controller.turn_control.remote("cancel_session", session_id=session, reason="shutdown")
            for session in list(loop.running)
        ], return_exceptions=True)
        errors.extend(result for result in cancelled if isinstance(result, Exception))
        # Cancel the parent gather once; a second child.cancel() could interrupt
        # the child's shielded sandbox cleanup before it finishes.
        for future in self.pending:
            future.cancel()
        await asyncio.gather(*self.pending, return_exceptions=True)
        await attempt(self._rollout_controller.turn_control.remote("drain", timeout=60.0))
        all_groups = list(self.pending.values()) + self.ready + self.raw
        for group in all_groups:
            await attempt(self._release(group))
        self.event("manager_closed", launched=self.launched, consumed=self.consumed,
              cleanup_errors=[type(error).__name__ for error in errors])
        self.pending.clear()
        self.ready.clear()
        self.raw.clear()
        if errors:
            raise ExceptionGroup("turn-level shutdown incomplete", errors)


class ManagerThread:
    """All manager objects are constructed and used on one running event loop."""
    def __init__(self, factory):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="turn-rollout")
        self.thread.start()
        async def build():
            return factory()
        try:
            self.manager = asyncio.run_coroutine_threadsafe(build(), self.loop).result()
        except BaseException:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            raise
        self.data_sampler = self.manager.data_sampler
        self.closed = False

    async def produce_batch(self, *args, **kwargs):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self.manager.produce_batch(*args, **kwargs), self.loop))

    async def save(self, *args, **kwargs):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self.manager.save(*args, **kwargs), self.loop))

    async def resume(self, *args, **kwargs):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self.manager.resume(*args, **kwargs), self.loop))

    def record_optimizer_outcome(self, train_step, updated):
        asyncio.run_coroutine_threadsafe(
            self.manager.record_optimizer_outcome(train_step, updated), self.loop).result()

    def record_weights_synced(self):
        asyncio.run_coroutine_threadsafe(self.manager.record_weights_synced(), self.loop).result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            asyncio.run_coroutine_threadsafe(self.manager.close(), self.loop).result(timeout=240)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)


class TurnManagerConfig(AgentLoopManagerConfig):
    """Opt-in persistent colocated manager; other manager modes are unchanged."""
    turn_async: TurnAsyncOptions = Field(default_factory=TurnAsyncOptions)

    @model_validator(mode="after")
    def single_runner(self):
        tasks = self.tasks if isinstance(self.tasks, list) else [self.tasks]
        if len(tasks) != 1:
            raise ValueError("turn-level colocated mode supports exactly one runner")
        loop_config = tasks[0].agent_loop_config
        if not isinstance(loop_config, TurnAgentInSandboxLoopConfig) or loop_config.cpu_resources is not None:
            raise ValueError("turn-level mode requires a local TurnAgentInSandboxLoopConfig")
        return self

    def build(self, *args, **kwargs):
        def factory():
            base = super(TurnManagerConfig, self).build(*args, **kwargs)
            return TurnManager(task_runners=base.task_runners, replay_buffer=base.replay_buffer,
                               rollout_controller=base._rollout_controller, logger=base.logger, options=self.turn_async)
        return ManagerThread(factory)
