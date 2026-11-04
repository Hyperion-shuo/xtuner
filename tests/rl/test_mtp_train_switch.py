"""CPU checks for disabling MTP loss without deleting rollout/checkpoint weights."""
from types import SimpleNamespace
import unittest

import torch

from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.rl.trainer.worker import TrainingWorker, WorkerConfig


class TestMTPTrainSwitch(unittest.TestCase):
    def make_worker(self, enabled, compose=True):
        mtp = MTPConfig(num_layers=4, share_weights=True,
                        detach_mtp_inputs=True, detach_mtp_lm_head_weight=True)
        model_cfg = Qwen3_5_VLMoE35BA3Config()
        model_cfg.text_config.mtp_config = mtp
        if not compose:
            model_cfg = model_cfg.text_config
        body = torch.nn.Parameter(torch.tensor([1.0]))
        draft = torch.nn.Parameter(torch.tensor([2.0]))
        parameters = [('layers.0.weight', body), ('mtp_block.layers.0.weight', draft)]
        cfg = SimpleNamespace(model_cfg=model_cfg, train_mtp=enabled)
        worker = object.__new__(TrainingWorker)
        worker._engine = SimpleNamespace(model=SimpleNamespace(trainable_parameters=lambda: parameters))
        TrainingWorker._init_mtp_training(worker, cfg)
        return worker, mtp, model_cfg, body, draft

    def test_default_keeps_mtp_training(self):
        self.assertTrue(WorkerConfig.model_fields['train_mtp'].default)
        for compose in (True, False):
            worker, mtp, _, _, draft = self.make_worker(True, compose)
            self.assertIs(worker.mtp_config, mtp)
            self.assertEqual(len(worker._mtp_trainable_parameters), 1)
            self.assertIs(worker._mtp_trainable_parameters[0], draft)

    def test_disabled_preserves_model_and_weights_for_sync(self):
        for compose in (True, False):
            worker, mtp, cfg, body, draft = self.make_worker(False, compose)
            self.assertIsNone(worker.mtp_config)
            self.assertEqual(worker._mtp_trainable_parameters, [])
            self.assertIs((cfg.text_config if compose else cfg).mtp_config, mtp)
            self.assertEqual(len(worker._engine.model.trainable_parameters()), 2)
            optimizer = torch.optim.AdamW([body, draft], lr=.01, weight_decay=.1)
            original = draft.detach().clone()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                body.square().sum().backward()
                optimizer.step()
            self.assertIsNone(draft.grad)
            self.assertNotIn(draft, optimizer.state)
            torch.testing.assert_close(draft, original, rtol=0, atol=0)
            self.assertLess(body.item(), 1.)


if __name__ == '__main__':
    unittest.main()
