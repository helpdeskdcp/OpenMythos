import os

import pytest
import torch

from open_mythos import OpenMythos, load_mythos_100m, mythos_100m, mythos_100m_v2

CHECKPOINT_PATH = "/root/openmythos/models/100m/mythos_100m_mixed_80k.pt"

# Tensor names/shapes taken directly from the released mythos_100m_mixed_80k.pt
# checkpoint (legacy GQA: wq/wk/wv/wo, not MLA's q_down/q_up/kv_down/kv_up).
EXPECTED_SHAPES = {
    "embed.weight": (199998, 512),
    "head.weight": (199998, 512),
    "freqs_cis": (256, 32),
    "freqs_cis_mla": (256, 16),
    "prelude.0.attn.wq.weight": (512, 512),
    "prelude.0.attn.wk.weight": (256, 512),
    "prelude.0.attn.wv.weight": (256, 512),
    "prelude.0.attn.wo.weight": (512, 512),
    "prelude.0.ffn.gate.weight": (682, 512),
    "prelude.0.ffn.up.weight": (682, 512),
    "prelude.0.ffn.down.weight": (512, 682),
    "recurrent.block.attn.wq.weight": (512, 512),
    "recurrent.block.ffn.router.weight": (4, 512),
    "recurrent.block.ffn.router_bias": (4,),
    "recurrent.block.ffn.routed_experts.0.gate.weight": (256, 512),
    "recurrent.block.ffn.routed_experts.0.down.weight": (512, 256),
    "recurrent.block.ffn.shared_experts.0.gate.weight": (512, 512),
    "recurrent.block.ffn.shared_experts.0.down.weight": (512, 512),
    "recurrent.injection.log_A": (512,),
    "recurrent.injection.log_dt": (1,),
    "recurrent.injection.B": (512,),
    "recurrent.act.halt.weight": (1, 512),
    "recurrent.act.halt.bias": (1,),
    "recurrent.lora.B": (8, 512),
    "recurrent.lora.down.weight": (8, 512),
    "recurrent.lora.scale.weight": (4, 8),
    "coda.1.attn.wk.weight": (256, 512),
    "coda.1.ffn.down.weight": (512, 682),
    "norm.weight": (512,),
}


def test_mythos_100m_config_is_gqa():
    cfg = mythos_100m()
    assert cfg.attn_type == "gqa"
    assert cfg.vocab_size == 199998
    assert cfg.dim == 512
    assert cfg.n_heads == 8
    assert cfg.n_kv_heads == 4
    assert cfg.max_loop_iters == 4
    assert cfg.prelude_layers == 2
    assert cfg.coda_layers == 2
    assert cfg.n_experts == 4
    assert cfg.n_shared_experts == 1
    assert cfg.n_experts_per_tok == 2
    assert cfg.expert_dim == 256
    assert cfg.lora_rank == 8


def test_mythos_100m_state_dict_matches_checkpoint_tensor_shapes():
    """Every tensor name/shape the model produces must equal the checkpoint's,
    without needing the (large, host-local) checkpoint file on disk."""
    model = OpenMythos(mythos_100m())
    state = model.state_dict()
    for name, shape in EXPECTED_SHAPES.items():
        assert name in state, f"missing key: {name}"
        assert tuple(state[name].shape) == shape, (
            f"{name}: expected {shape}, got {tuple(state[name].shape)}"
        )


def test_mythos_100m_v2_grows_non_embedding_core():
    """v2 keeps the same vocab/attention skeleton but a much larger MoE core,
    and is intentionally NOT checkpoint-compatible with mythos_100m()."""
    cfg = mythos_100m_v2()
    assert cfg.vocab_size == 199998
    assert cfg.attn_type == "gqa"
    assert cfg.n_experts == 8
    assert cfg.expert_dim == 4096

    model = OpenMythos(cfg)
    total = sum(p.numel() for p in model.parameters())
    non_embed = total - model.embed.weight.numel()
    assert total == 173_455_906
    assert non_embed == 71_056_930

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(input_ids, n_loops=cfg.max_loop_iters)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH),
    reason="real 100M checkpoint not present on this host",
)
def test_mythos_100m_checkpoint_loads_strict_and_generates():
    model = load_mythos_100m(CHECKPOINT_PATH, device="cpu")

    total = sum(p.numel() for p in model.parameters())
    assert total == 112_898_594

    input_ids = torch.randint(0, 199998, (1, 8))
    with torch.no_grad():
        logits = model(input_ids, n_loops=4)
    assert logits.shape == (1, 8, 199998)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    out = model.generate(input_ids, max_new_tokens=8, n_loops=4, top_k=50)
    assert out.shape == (1, 16)
    assert not torch.isnan(out.float()).any()
