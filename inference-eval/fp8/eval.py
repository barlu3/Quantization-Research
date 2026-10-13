import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import run

# shortcut: MPS has no float8 dtype ("Undefined type Float8_e4m3fn"), so this is weight-only fake
# quantization: weights are rounded through fp8 on CPU, then stored and multiplied in bf16 on the GPU.
# Measures fp8's accuracy cost, not its speed/memory; upgrade to a real fp8 kernel on hardware that has one.


def quantize_fp8(w: torch.Tensor) -> torch.Tensor:
    """Round a [out_features, in_features] bf16 weight through torch.float8_e4m3fn; return it dequantized as bf16."""
    # pick a scale so w fits e4m3's range (max finite = 448), cast to fp8, cast back, undo the scale.
    biggest = w.abs().max()
    scale = biggest.float() / 448
    rounded = (w.float()/ scale).to(torch.float8_e4m3fn)

    return (rounded.float() * scale ).to(torch.bfloat16)



def fp8_weights(model):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name != "lm_head":  # lm_head is usually kept high-precision
            q = quantize_fp8(m.weight.data)
            # e4m3 has 3 mantissa bits, so honest rounding stays well under this; ~1.0 means values underflowed to 0
            assert (q - m.weight.data).float().norm() / m.weight.data.float().norm() < 0.05, name
            m.weight.data = q
    return model


run("fp8", torch.bfloat16, fp8_weights)
