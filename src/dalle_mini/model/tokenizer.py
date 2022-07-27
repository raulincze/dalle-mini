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
        assert return_tensors == "np"
        max_coords = COORD_EMBED_SIZE if truncation else max_length
        np_coords = np.zeros((len(inputs), max_coords))
        for i, input in enumerate(inputs):
            coordinates = json.loads(input)
            np_coords[i, 0] = coordinates["position"]["x"]
            np_coords[i, 1] = coordinates["position"]["y"]
            np_coords[i, 2] = coordinates["position"]["z"]
            np_coords[i, 3] = coordinates["rotation"]["x"]
            np_coords[i, 4] = coordinates["rotation"]["y"]
            np_coords[i, 5] = coordinates["rotation"]["z"]
            np_coords[i, 6] = coordinates["rotation"]["w"]
        return {
            'input_ids': np_coords,
            'attention_mask': np.array(range(COORD_EMBED_SIZE)),
            'length': COORD_EMBED_SIZE
        }
