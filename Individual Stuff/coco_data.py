from __future__ import annotations

import json
import pickle

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Loads and organizes the COCO dataset. Also loads 512-dimensional ResNet descriptor vectors for each image

class COCOData:

    def __init__(self, coco_json_path: str | Path, resnet_features_path: str | Path):

        with Path(coco_json_path).open("r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Load image-ID -> ResNet descriptor mapping
        with Path(resnet_features_path).open("rb") as f:
            self.resnet_features = pickle.load(f)

        # Only use images that have ResNet descriptors
        self.valid_image_ids = set(self.resnet_features.keys())

        # Links image id with the actual image
        self.image_id_to_url: Dict[int, str] = {}
        
        # Links image with its captions
        self.image_id_to_caption_ids: Dict[int, List[int]] = defaultdict(list)

        # Links captions with its images
        self.caption_id_to_image_id: Dict[int, int] = {}
        
        # Links caption id with what it actually says
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

        # cache reshaped descriptor arrays so repeated lookups are free
        self._descriptor_cache: Dict[int, np.ndarray] = {}

    def split_image_ids(self, train_fraction: float = 0.8, seed: int = 0):
        rng = np.random.default_rng(seed)

        shuffled = self.image_ids.copy()
        rng.shuffle(shuffled)

        split_index = int(len(shuffled) * train_fraction)

        train_ids = shuffled[:split_index]
        validation_ids = shuffled[split_index:]

        return train_ids, validation_ids

    def get_descriptor(self, image_id: int):
        cached = self._descriptor_cache.get(image_id)
        if cached is not None:
            return cached

        descriptor = np.asarray(
            self.resnet_features[image_id],
            dtype=np.float32,
        ).reshape(-1)
        self._descriptor_cache[image_id] = descriptor
        return descriptor
        
    def get_descriptors(self, image_ids: np.ndarray | list[int]):
        descriptors = []

        for id in image_ids:
            descriptor = self.get_descriptor(int(id))
            descriptors.append(descriptor)

        return np.stack(descriptors)
    
    def get_caption(self, caption_id: int):
        return self.caption_id_to_text[caption_id]
    
    def get_captions(self, image_id: int): #get the correct captions for the corresponding image
        captions = []

        for caption_id in self.get_caption_ids(image_id):
            caption_text = self.get_caption(caption_id)
            captions.append(caption_text)
        return captions

    def get_all_captions(self):
        all_captions = []
        
        for caption_id in self.caption_ids:
            caption_text = self.get_caption(int(caption_id))
            all_captions.append(caption_text)
            
        return all_captions
            
    
    def get_image_id(self, caption_id: int): 
        return self.caption_id_to_image_id[caption_id]

    def get_image_url(self, image_id: int):
        return self.image_id_to_url[image_id]

    def get_caption_ids(self, image_id: int):
        return self.image_id_to_caption_ids[image_id]


