"""Persistent task-neutral sandbox Trials with turn trace validation."""
from __future__ import annotations
import asyncio
import math
import uuid
from pydantic import Field
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop_manager.turn_async_config import TurnTrialOptions
from xtuner.v1.rl.rollout.trace_store import get_store
from .agent_in_sandbox_loop import AgentInSandboxLoop, AgentInSandboxLoopConfig
from .schemas import RolloutStatus

def validate_segment(sample, *, require_routing=False):
    ids, labels, probabilities = sample.input_ids, sample.labels, sample.logprobs
    if not ids or not labels or not probabilities or not len(ids) == len(labels) == len(probabilities):
        raise ValueError("input/label/logprob lengths differ")
    expected = [token for token,label in zip(ids[1:], labels[1:]) if label != -100]
    if not expected or sample.response_ids != expected:
        raise ValueError("Response tokens do not match selected loss positions")
    if any(label != -100 and label != token for token,label in zip(ids,labels)):
        raise ValueError("A selected label differs from its original token")
    if not all(math.isfinite(p) for p in probabilities):
        raise ValueError("Nonfinite behavior logprob")
    versions = sample.extra_fields.get("response_policy_versions", [])
    steps = sample.response_model_steps
    if not steps or len(steps) != len(expected) or len(versions) != len(expected):
        raise ValueError("Generation versions do not cover loss tokens")
    if any(v is None for v in [*versions, *steps]):
        raise ValueError("Missing original generation version")
    if not sample.extra_fields.get("routing_source_request_id") or (require_routing and sample.routed_experts is None):
        raise ValueError("Missing routing source identity")
    return {"input_tokens":len(ids), "loss_tokens":len(expected),
            "policy_versions":sorted(set(versions)), "source_optimizer_steps":sorted(set(steps))}


class TurnAgentInSandboxLoop(AgentInSandboxLoop):
    def __init__(self, *, options=None, **kwargs):
        super().__init__(**kwargs)
        self.options = options or TurnTrialOptions()
        self.event, self.disposition = self.options.event, self.options.disposition
        self.running = {}
        self.trial_admission = asyncio.Event()
        self.sampling_window = self.policy_version = None

    def set_generation_context(self, sampling_window, policy_version):
        self.sampling_window, self.policy_version = sampling_window, policy_version

    def pause_new_trials(self): self.trial_admission.clear()
    def resume_new_trials(self): self.trial_admission.set()

    async def generate_group(self, states, **kwargs):
        if len({s.extra_fields["rollout_item"].id for s in states}) != 1:
            raise ValueError("A prompt group cannot contain different tasks")

        async def one(state):
            async with self._sample_semaphore:
                while not self.trial_admission.is_set():
                    await self.trial_admission.wait()
                state.session_id = uuid.uuid4().int
                session = str(state.session_id)
                item = state.extra_fields["rollout_item"]
                identity = dict(run_id=self.options.run_id, task_id=item.id, data_source=item.data_source,
                                rollout_id=state.rollout_id, attempt_id=uuid.uuid4().hex, session_id=session,
                                group_id=state.group_id, sampling_window=self.sampling_window,
                                policy_version=self.policy_version)
                state.extra_fields["trial"] = identity
                item.metadata["trial"] = identity
                if self.options.identity_adapter:
                    self.options.identity_adapter(state, item, identity)
                self.running[session] = asyncio.current_task()
                try:
                    if self.options.before_trial:
                        await self.options.before_trial(state, item, identity)
                    state.sample_params = self.sample_params
                    result = await self.generate_sample(state)
                    failed = [s for s in result if s.status == Status.FAILED]
                    if failed:
                        self.disposition([failed], filtered=True, filter_reason="rollout_or_trace_failed")
                    return result
                except asyncio.CancelledError:
                    state.status, state.reward = Status.FAILED, None
                    state.error_msg = "turn_async_cancelled"
                    self.disposition([[state]], filtered=True, filter_reason="cancelled")
                    return [state]
                finally:
                    self.running.pop(session, None)
        tasks = [asyncio.create_task(one(state)) for state in states]
        try:
            groups = await asyncio.gather(*tasks)
            return [sample for group in groups for sample in group]
        except BaseException:
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _build_rollout_states(self, state, item):
        if self.options.result_adapter:
            item = self.options.result_adapter(item)
        contract = item.artifacts.get("trial_result", {})
        if item.status == RolloutStatus.COMPLETED:
            if not contract.get("cleanup_ok", False) or contract.get("verifier_status") != "completed":
                raise ValueError("Completed Trial requires real scoring and confirmed cleanup")
            if any(not msg.get("request_id") for msg in item.artifacts.get("messages", [])):
                raise ValueError("Turn trace requires request identity")
        result = await super()._build_rollout_states(state, item)
        for sample in result:
            sample.extra_fields.update(raw_reward=contract.get("raw_reward"),
                                       verifier_status=contract.get("verifier_status"), cleanup_ok=contract.get("cleanup_ok", False))
        if item.status == RolloutStatus.COMPLETED:
            for sample in result:
                alignment = validate_segment(sample, require_routing=self.options.require_routing)
                self.event("trace_validated", **sample.extra_fields["trial"], **alignment,
                      routing_source_request_id=sample.extra_fields["routing_source_request_id"])
            await get_store().retain_turn_routes.remote(str(state.session_id), [
                sample.extra_fields["routing_source_request_id"] for sample in result])
        return result


class TurnAgentInSandboxLoopConfig(AgentInSandboxLoopConfig):
    turn_trial: TurnTrialOptions = Field(default_factory=TurnTrialOptions)

    def build_local(self, rollout_controller=None, judger=None, logger=None):
        return TurnAgentInSandboxLoop(rollout_ctl=rollout_controller, hf_checkpoint=self.hf_checkpoint,
                         sample_params=self.sample_params, judger=judger, logger=logger,
                         max_concurrent_samples=self.max_concurrent_samples, mode=self.mode, options=self.turn_trial)
