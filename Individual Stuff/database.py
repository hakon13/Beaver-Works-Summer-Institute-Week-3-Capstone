from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from coco_data import COCOData
from image_to_caption_model import ImageToCaption

class ImageDatabase:
    def __init__(self, model: ImageToCaption, coco: COCOData):
        self.model = model
        self.coco = coco
        self.image_ids = []
        self.embeddings = []

    def build(self):
        image_ids = self.coco.image_ids
        descriptors = self.coco.get_descriptors(image_ids)

        self.model.eval()
        with torch.no_grad():
            descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32)
            out = self.model(descriptors_tensor)

        self.image_ids = np.array(image_ids)
        self.embeddings = out.numpy()
        print("Databse built with", len(self.image_ids), "images")

    def query(self, caption_text, vectorizer, k=5):
        w = vectorizer(caption_text)

        sim = self.embeddings @ w

        best = np.argsort(sim)
        best = best[-k:]
        best = best[::-1]

        results = []
        for i in best:
            image_id = int(self.image_ids[i])
            url = self.coco.get_image_url(image_id)
            score = float(sim[i])
            results.append((image_id, url, score))
        return results

    def save(self, path):
        f = open(path, "wb")
        pickle.dump([self.image_ids, self.embeddings], f)
        f.close()

    def load(self, path):
        f = open(path, "rb")
        stuff = pickle.load(f)
        self.image_ids = stuff[0]
        self.embeddings = stuff[1]
        f.close()