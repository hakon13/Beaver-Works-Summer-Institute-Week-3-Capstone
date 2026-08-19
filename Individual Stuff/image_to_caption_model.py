import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coco_data import COCOData
from captions_to_vec import CaptionVectorizer

class ImageToCaption(nn.Module):
    def __init__(self, in_features: int = 512, out_features: int = 200):

        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias = False)

        nn.init.xavier_normal_(self.linear.weight)
    
    def forward(self, descriptors: torch.Tensor) -> torch.Tensor: 

        embedded = self.linear(descriptors)
        normalized = F.normalize(embedded, p=2, dim=1)

        return normalized


def loss(model: "ImageToCaption", coco: COCOData, vectorizer: CaptionVectorizer, triplets: list): #triplets should be a (N, 3) array where N is the batch size
    w_true_captions = []
    w_confusor_captions = []
    d_true_images = []

    for triplet in triplets:

        correct_caption_id = triplet[0]
        correct_image_id = triplet[1]
        wrong_caption_id = triplet[2]

        true_caption_text = coco.get_caption(correct_caption_id)
        confusor_caption_text = coco.get_caption(wrong_caption_id)

        w_true_caption = vectorizer(true_caption_text)
        w_confusor_caption = vectorizer(confusor_caption_text)

        d_true_image = coco.get_descriptor(correct_image_id)

        w_true_captions.append(w_true_caption)
        w_confusor_captions.append(w_confusor_caption)
        d_true_images.append(d_true_image)

    w_true_captions = torch.tensor(np.stack(w_true_captions), dtype=torch.float32)
    w_confusor_captions = torch.tensor(np.stack(w_confusor_captions), dtype=torch.float32)
    d_true_images = torch.tensor(np.stack(d_true_images), dtype=torch.float32)

    # embed the true images with the (trainable) model
    w_true_images = model(d_true_images)

    sim_true = torch.sum(w_true_images * w_true_captions, dim=1)
    sim_confusor = torch.sum(w_true_images * w_confusor_captions, dim=1)

    target = torch.ones_like(sim_true)

    return F.margin_ranking_loss(sim_true, sim_confusor, target, margin=0.25)

