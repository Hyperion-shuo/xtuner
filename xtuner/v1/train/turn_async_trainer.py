"""Colocated training bridge for persistent turn-boundary agent rollouts."""
from __future__ import annotations
import math
from pydantic import Field, model_validator
from xtuner.v1.rl.agent_loop_manager.turn_async import TurnManagerConfig
from xtuner.v1.rl.agent_loop_manager.turn_async_config import TurnTrainingOptions
from .rl_trainer import RLColocateTrainer, RLColocateTrainerConfig, is_valid_for_training


def optimizer_update_succeeded(worker_logs, skip_grad_norm_threshold):
    """Read the real worker optimizer decision; missing/conflicting evidence fails closed.

    This mode currently requires one optimizer step per rollout batch. Metrics
    from several steps cannot identify which weights were actually changed.
    """
    if not worker_logs:
        return None
    outcomes = set()
    for worker in worker_logs:
        metrics = worker.get('train_metrics', [])
        if (len(metrics) != 1 or isinstance(metrics[0].get('grad_norm'), bool)
                or not isinstance(metrics[0].get('grad_norm'), (int, float))):
            return None
        norm = metrics[0]['grad_norm']
        outcomes.add(math.isfinite(norm) and (skip_grad_norm_threshold is None or norm <= skip_grad_norm_threshold))
    return outcomes.pop() if len(outcomes) == 1 else None


class TurnRLColocateTrainer(RLColocateTrainer):
    def __init__(self, cfg):
        self.turn_training = cfg.turn_training
        super().__init__(cfg)

    def _sync_weights_and_save(self, train_step, step_timer_dict):
        synced = super()._sync_weights_and_save(train_step, step_timer_dict)
        if synced:
            self.agent_loop_manager.record_weights_synced()
        self.turn_training.event('weight_sync_completed', train_step=train_step, weights_synced=bool(synced))
        return synced

    def _prepare_train_data(self, data_groups, *args, **kwargs):
        valid = [g for g in data_groups if is_valid_for_training(g, self.logger)]
        invalid = [g for g in data_groups if not any(g is v for v in valid)]
        result, info = super()._prepare_train_data(data_groups, *args, **kwargs)
        self.turn_prepared = valid
        self.turn_training.disposition(invalid, filtered=True, filter_reason='training_validity_check')
        if self.turn_training.prepared_metrics:
            info.update(self.turn_training.prepared_metrics(valid))
        return result, info

    def _train_one_batch(self, train_batch, train_step, step_timer_dict, **kwargs):
        self.turn_prepared = None
        if self.turn_training.before_train:
            self.turn_training.before_train(self, train_step)
        info = super()._train_one_batch(train_batch, train_step, step_timer_dict, **kwargs)
        if self.turn_prepared is None:
            raise RuntimeError('Missing prepared batch evidence')
        updated = optimizer_update_succeeded(info['workers_log_item'],
                                             self._train_worker_cfg.optim_cfg.skip_grad_norm_threshold)
        outcome = 'updated' if updated is True else 'skipped' if updated is False else 'unknown'
        self.turn_training.disposition(self.turn_prepared, consumed=updated, training_step=train_step,
                                       training_outcome=outcome)
        self.turn_training.event('optimizer_outcome', train_step=train_step, outcome=outcome)
        if self.turn_training.after_train:
            self.turn_training.after_train(self, train_step, info, outcome)
        self.agent_loop_manager.record_optimizer_outcome(train_step, updated)
        return info

    def _release_trace_sessions_after_train_batch(self, train_batch):
        # Other pending/ready groups still own traces; never release all here.
        self._release_trace_sessions([str(s.session_id) for g in train_batch for s in g if s.session_id is not None])

    def _fit(self):
        try:
            return super()._fit()
        finally:
            self.agent_loop_manager.close()


class TurnRLColocateTrainerConfig(RLColocateTrainerConfig):
    turn_training: TurnTrainingOptions = Field(default_factory=TurnTrainingOptions)

    @model_validator(mode='after')
    def validate_turn_mode(self):
        if self.train_worker_cfg.optimizer_steps != 1:
            raise ValueError("turn mode currently requires optimizer_steps=1 for unambiguous update evidence")
        if not isinstance(self.agent_loop_manager_cfg, TurnManagerConfig):
            raise ValueError('TurnRLColocateTrainer requires TurnManagerConfig')
        if not self.rollout_config.turn_level_async:
            raise ValueError('TurnRLColocateTrainer requires rollout_config.turn_level_async=True')
        return self

    def build(self):
        return TurnRLColocateTrainer(self)
