from __future__ import annotations

import json
import pickle

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class COCOData:

    def __init__(self, coco_json_path: str | Path, resnet_features_path: str | Path):
        coco_json_path = Path(coco_json_path)
        resnet_features_path = Path(resnet_features_path)

        with coco_json_path.open("r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Load image-ID -> ResNet descriptor mapping
        with resnet_features_path.open("rb") as f:
            self.resnet_features = pickle.load(f)

        # Only use images that have ResNet descriptors
        self.valid_image_ids = set(self.resnet_features.keys())

        self.image_id_to_url: Dict[int, str] = {}
        self.image_id_to_caption_ids: Dict[int, List[int]] = defaultdict(list)

        self.caption_id_to_image_id: Dict[int, int] = {}
        self.caption_id_to_text: Dict[int, str] = {}

        # Store image information
        for image_info in coco_data["images"]:
            image_id = image_info["id"]

            if image_id not in self.valid_image_ids:
                continue

            self.image_id_to_url[image_id] = image_info["coco_url"]

        # Store captions
        for annotation in coco_data["annotations"]:
            image_id = annotation["image_id"]

            if image_id not in self.valid_image_ids:
                continue

            caption_id = annotation["id"]
            caption = annotation["caption"]

            self.caption_id_to_image_id[caption_id] = image_id
            self.caption_id_to_text[caption_id] = caption
            self.image_id_to_caption_ids[image_id].append(caption_id)

        self.image_ids = np.array(sorted(self.image_id_to_url.keys()), dtype=np.int64)

        self.caption_ids = np.array(sorted(self.caption_id_to_text.keys()), dtype=np.int64)

    def get_descriptor(self, image_id: int) -> np.ndarray:
        return np.asarray(
            self.resnet_features[image_id],
            dtype=np.float32,
        ).reshape(-1)

    def get_caption(self, caption_id: int) -> str:
        return self.caption_id_to_text[caption_id]

    def get_image_url(self, image_id: int) -> str:
        return self.image_id_to_url[image_id]

    def get_caption_ids(self, image_id: int) -> List[int]:
        return self.image_id_to_caption_ids[image_id]

    def split_image_ids(
        self,
        train_fraction: float = 0.8,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:

        rng = np.random.default_rng(seed)

        shuffled = self.image_ids.copy()
        rng.shuffle(shuffled)

        split_index = int(len(shuffled) * train_fraction)

        train_ids = shuffled[:split_index]
        validation_ids = shuffled[split_index:]

        return train_ids, validation_ids

def generate_bad_captions(caption_id):
    COCOData.get_caption_ids('wait so did Newton just do this?',caption_id)
generate_bad_captions(1)