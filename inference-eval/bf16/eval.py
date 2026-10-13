import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import run

run("bf16", torch.bfloat16)
