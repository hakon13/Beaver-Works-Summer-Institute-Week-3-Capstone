from typing import Sequence

import numpy as np
import torch

from image_to_caption_model import loss

from coco_data import COCOData


# triplets: (true-caption-ID, true-image-ID, confusor-caption-ID)

def generate_triplets(coco: COCOData, allowed_image_ids: Sequence[int], num_triplets: int, seed: int = 0):

    rng = np.random.default_rng(seed)

    allowed_image_ids = np.asarray(allowed_image_ids, dtype=np.int64)

    if len(allowed_image_ids) < 2:
        raise ValueError("At least two images are required to create triplets.")

    valid_true_image_ids = [image_id for image_id in allowed_image_ids if len(coco.get_caption_ids(int(image_id))) > 0]

    triplets = []

    for _ in range(num_triplets):
        correct_image_id = int(rng.choice(valid_true_image_ids))

        correct_caption = int(rng.choice(coco.get_caption_ids(correct_image_id)))

        wrong_image_id = correct_image_id

        while wrong_image_id == correct_image_id:
            wrong_image_id = int(rng.choice(valid_true_image_ids))
            
        wrong_caption = int(rng.choice(coco.get_caption_ids(wrong_image_id)))

        triplets.append((correct_caption, correct_image_id, wrong_caption))

    return triplets

def train(model, train_ids, coco: COCOData, vectorizer, epochs: int = 1, batch_size: int = 32):
    train_ids = np.asarray(train_ids)

    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    model.train()
    for epoch_cnt in range(epochs):
        idxs = np.arange(len(train_ids))
        np.random.shuffle(idxs)

        for batch_cnt in range(0, len(train_ids) // batch_size):
            batch_indices = idxs[(batch_cnt * batch_size):((batch_cnt + 1) * batch_size)]
            batch = train_ids[batch_indices]

            triplets = generate_triplets(coco=coco, allowed_image_ids=batch, num_triplets=len(batch), seed=np.random.randint(1000000))
            loss_val = loss(model=model, coco=coco, vectorizer=vectorizer, triplets=triplets)

            optim.zero_grad()
            loss_val.backward()
            optim.step()

    return None


