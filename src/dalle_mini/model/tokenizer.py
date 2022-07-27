from typing import List
import json
""" DalleBart tokenizer """
from transformers import BartTokenizerFast

from .utils import PretrainedFromWandbMixin
import numpy as np


class DalleBartTokenizer(PretrainedFromWandbMixin, BartTokenizerFast):
    pass


COORD_EMBED_SIZE = 7


class GameCoordinateTokenizer():
    def __call__(
        self,
        inputs: List[str],
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "np",
    ):
        print("!!!!!!!!!!!!!!!!!! Max len is", max_length)
        assert return_tensors == "np"
        max_coords = COORD_EMBED_SIZE if truncation else max_length
        tokenized = {
            'input_ids': np.zeros((len(inputs), max_coords)),
            'attention_mask': np.zeros((len(inputs), max_coords)),
            'length': np.array([COORD_EMBED_SIZE] * len(inputs)),
        }
        for i, input in enumerate(inputs):
            coordinates = json.loads(input)
            tokenized['input_ids'][i, 0] = coordinates["position"]["x"]
            tokenized['input_ids'][i, 1] = coordinates["position"]["y"]
            tokenized['input_ids'][i, 2] = coordinates["position"]["z"]
            tokenized['input_ids'][i, 3] = coordinates["rotation"]["x"]
            tokenized['input_ids'][i, 4] = coordinates["rotation"]["y"]
            tokenized['input_ids'][i, 5] = coordinates["rotation"]["z"]
            tokenized['input_ids'][i, 6] = coordinates["rotation"]["w"]
            tokenized['attention_mask'][i, :7] = np.ones(COORD_EMBED_SIZE)
        print("!!!!!!!!!!!!!!!!! Shape is", tokenized['attention_mask'].shape)
        return tokenized
