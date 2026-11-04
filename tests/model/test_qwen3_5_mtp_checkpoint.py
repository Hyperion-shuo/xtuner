"""CPU checkpoint coverage; no model forward, CUDA kernel, or transport is mocked in as passing.

The small module uses the production HF loader, LoadSpec, exporter, and layout
adapters. It exercises both the original Qwen3.6 format and legacy checkpoints
whose trunk experts are fused but MTP experts are split.
"""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from xtuner.v1.model.base import BaseModel
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE, Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.utils.load_spec import LoadSpec, ShardDescriptor


@pytest.fixture(autouse=True)
def cpu_checkpoint_runtime(monkeypatch):
    # The checkpoint API also profiles device memory and clears accelerator
    # caches. CPU has no such allocator; leave all tensor I/O and layout code
    # real and replace only those runtime bookkeeping calls.
    monkeypatch.setattr("xtuner.v1.model.base.DEVICE", "cpu")
    monkeypatch.setattr("xtuner.v1.model.base.DEVICE_MODULE", SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setattr("xtuner.v1.model.base.profile_time_and_memory", lambda _: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)


def _module(**children):
    module = nn.Module()
    for name, child in children.items():
        module.add_module(name, child)
    return module


def _experts():
    return _module(
        fused_w1w3=nn.Linear(6, 4 * 2 * 3, bias=False, dtype=torch.bfloat16),
        fused_w2=nn.Linear(3, 4 * 6, bias=False, dtype=torch.bfloat16),
    )


class _CheckpointModel(Qwen3_5_VLTextMoE):
    """Only the parameter tree is small; checkpoint code is inherited unchanged."""

    def __init__(self, mtp_format):
        config = Qwen3_5_VLTextMoE35BA3BConfig(
            compile_cfg=False,
            n_routed_experts=4,
            hidden_size=6,
            moe_intermediate_size=3,
            mtp_hf_expert_format=mtp_format,
        )
        BaseModel.__init__(self, config)
        self.layers = nn.ModuleDict({"0": _module(experts=_experts())})
        self.mtp_block = _module(layers=nn.ModuleList([_module(decoder_layer=_module(experts=_experts()))]))
        self._init_load_spec()


def _checkpoint(mtp_format):
    checkpoint = {}
    for offset, prefix in enumerate(("model.language_model.layers.0.mlp.experts", "mtp.layers.0.mlp.experts")):
        # Distinct, exactly representable values catch expert and gate/up swaps.
        gate_up = torch.arange(4 * 6 * 6, dtype=torch.bfloat16).reshape(4, 6, 6) + offset * 64
        down = torch.arange(4 * 6 * 3, dtype=torch.bfloat16).reshape(4, 6, 3) - offset * 64
        if prefix.startswith("mtp.") and mtp_format == "split":
            for expert in range(4):
                checkpoint[f"{prefix}.{expert}.gate_proj.weight"] = gate_up[expert, :3].clone()
                checkpoint[f"{prefix}.{expert}.up_proj.weight"] = gate_up[expert, 3:].clone()
                checkpoint[f"{prefix}.{expert}.down_proj.weight"] = down[expert].clone()
        else:
            checkpoint[f"{prefix}.gate_up_proj"] = gate_up
            checkpoint[f"{prefix}.down_proj"] = down
    return checkpoint


@pytest.mark.parametrize("mtp_format", ["fused", "split"])
def test_load_export_and_reload_with_fused_trunk(tmp_path, mtp_format):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = _checkpoint(mtp_format)
    save_file(checkpoint, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    model = _CheckpointModel(mtp_format)
    loaded, unloaded, missing = model.from_hf(source, strict=True)
    assert loaded == set(model.state_dict())
    assert not unloaded and not missing

    fused_reference = _checkpoint("fused")
    for parameter_prefix, hf_prefix in (
        ("layers.0.experts", "model.language_model.layers.0.mlp.experts"),
        ("mtp_block.layers.0.decoder_layer.experts", "mtp.layers.0.mlp.experts"),
    ):
        actual_gate_up = model.state_dict()[f"{parameter_prefix}.fused_w1w3.weight"]
        actual_down = model.state_dict()[f"{parameter_prefix}.fused_w2.weight"]
        for expert in range(4):
            # Check independent expert slices, including gate/up ordering.
            torch.testing.assert_close(
                actual_gate_up[expert * 6 : expert * 6 + 3],
                fused_reference[f"{hf_prefix}.gate_up_proj"][expert, :3],
            )
            torch.testing.assert_close(
                actual_gate_up[expert * 6 + 3 : (expert + 1) * 6],
                fused_reference[f"{hf_prefix}.gate_up_proj"][expert, 3:],
            )
            torch.testing.assert_close(
                actual_down[expert * 6 : (expert + 1) * 6], fused_reference[f"{hf_prefix}.down_proj"][expert]
            )

    # This is the same production iterator used by rollout weight updates.
    exported = {}
    for names, values in model._get_hf_param(model._load_spec_params(), dtype=torch.bfloat16, device="cpu"):
        exported.update(zip(names, values, strict=True))
    assert exported.keys() == checkpoint.keys()
    for name, expected in checkpoint.items():
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)

    destination = tmp_path / "saved"
    model.save_hf(destination)
    index = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
    assert index.keys() == checkpoint.keys()
    for name, filename in index.items():
        torch.testing.assert_close(load_file(destination / filename)[name], checkpoint[name], rtol=0, atol=0)
    reloaded = _CheckpointModel(mtp_format)
    assert reloaded.from_hf(destination, strict=True)[1:] == (set(), set())
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mtp_format", ["fused", "split"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("interleave_factor", [1, 4])
@pytest.mark.parametrize("projection", ["fused_w1w3", "fused_w2"])
def test_load_plan_keeps_expert_slices(mtp_format, rank, interleave_factor, projection):
    model = _CheckpointModel(mtp_format)
    name = f"mtp_block.layers.0.decoder_layer.experts.{projection}.weight"
    global_shape = tuple(model.state_dict()[name].shape)
    # A pure layout test: emulate group metadata only, not distributed collectives.
    shard = ShardDescriptor.model_construct(dim=0, group=object(), interleave_factor=interleave_factor)
    with (
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=2),
    ):
        spec = LoadSpec(
            name=name,
            global_hf_keys=model.to_hf_key_list(name),
            global_shape=global_shape,
            fused_dim=0 if mtp_format == "split" else None,
            shards=[shard],
        )
        plan = spec.plan_hf_load()
    local = torch.empty(plan.target_shape, dtype=torch.bfloat16)
    checkpoint = _checkpoint(mtp_format)
    plan.load_into([checkpoint[key] for key in plan.hf_keys], local, model.hf_tensor_to_canonical)
    suffix = "gate_up_proj" if projection == "fused_w1w3" else "down_proj"
    full = _checkpoint("fused")[f"mtp.layers.0.mlp.experts.{suffix}"].reshape(global_shape)
    # Independent reference for the rank's consecutive runs in global row order.
    chunks = full.chunk(2 * interleave_factor, dim=0)
    expected = torch.cat([chunks[i * 2 + rank] for i in range(interleave_factor)], dim=0)
    torch.testing.assert_close(local, expected, rtol=0, atol=0)


def test_legacy_default_and_explicit_fused_format():
    assert Qwen3_5_VLTextMoE35BA3BConfig().mtp_hf_expert_format == "split"
    with pytest.raises(ValueError):
        Qwen3_5_VLTextMoE35BA3BConfig(mtp_hf_expert_format="auto")


@pytest.mark.parametrize("checkpoint_format", ["fused", "split"])
def test_wrong_explicit_format_fails_strict_loading(tmp_path, checkpoint_format):
    save_file(_checkpoint(checkpoint_format), tmp_path / "model.safetensors")
    model = _CheckpointModel("split" if checkpoint_format == "fused" else "fused")
    with pytest.raises(RuntimeError, match="Missing keys in checkpoint"):
        model.from_hf(tmp_path, strict=True)


@pytest.mark.parametrize("mtp_format", ["fused", "split"])
def test_parameter_update_reaches_hf_export(tmp_path, mtp_format):
    checkpoint = _checkpoint(mtp_format)
    save_file(checkpoint, tmp_path / "model.safetensors")
    model = _CheckpointModel(mtp_format)
    model.from_hf(tmp_path)
    # A synthetic CPU update checks that the export sees current parameters,
    # not cached source weights. This does not exercise the RL loss or training.
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    sum(parameter.float().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    updated = {}
    for names, values in model._get_hf_param(model._load_spec_params(), dtype=torch.bfloat16, device="cpu"):
        updated.update(zip(names, values, strict=True))
    assert updated.keys() == checkpoint.keys()
    for key, original in checkpoint.items():
        torch.testing.assert_close(updated[key], original - 1, rtol=0, atol=0)


@pytest.mark.parametrize("mtp_format", ["fused", "split"])
def test_shared_four_depth_mtp_has_one_physical_layer(mtp_format):
    config = Qwen3_5_VLTextMoE35BA3BConfig(
        compile_cfg=False,
        ep_size=1,
        dispatcher=None,
        mtp_hf_expert_format=mtp_format,
        mtp_config=MTPConfig(num_layers=4, share_weights=True),
    )
    # The builder creates an offload stream even on meta. Mock only that handle.
    with patch("torch.cuda.Stream"), torch.device("meta"):
        model = config.build()
    assert len(model.mtp_block.layers) == 1
    assert model.mtp_block.mtp_config.num_layers == 4
    for name, parameter in model.mtp_block.named_parameters():
        if "experts.fused_w1w3.weight" in name:
            assert tuple(parameter.shape) == (256 * 1024, 2048)
            assert model.to_hf_key_list(f"mtp_block.{name}")[0].startswith("mtp.layers.0.mlp.experts.")
        if "experts.fused_w2.weight" in name:
            assert tuple(parameter.shape) == (256 * 2048, 512)
