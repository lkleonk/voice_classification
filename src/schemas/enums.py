from __future__ import annotations

from enum import Enum, auto


class RunMode(Enum):
    # train from scratch; always save model info
    NORMAL_TRAIN_ALWAYS_W_RESULTS = auto()
    # train from scratch; if results are good, save model info
    NORMAL_TRAIN_ALWAYS_WITHOUT_RESULTS = auto()
    # fine-tune a pretrained model
    LOAD_FULLY_PRETRAINED_MODEL = auto()
