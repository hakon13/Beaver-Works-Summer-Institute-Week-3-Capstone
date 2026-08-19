import pickle
import torch
from pathlib import Path



from coco_data import COCOData
from captions_to_vec import CaptionVectorizer
from image_to_caption_model import ImageToCaption, loss
from training import generate_triplets, train
from database import ImageDatabase
from frontend import create_frontend



DATA_DIR = Path(__file__).parent / "data"
COCO_JSON_PATH = DATA_DIR / "captions_train2014.json"
RESNET_FEATURES_PATH = DATA_DIR / "resnet18_features.pkl"
GLOVE_PATH = DATA_DIR / "glove.6B.200d.txt.w2v"
MODEL_WEIGHTS_PATH = ""


print("loading data...")
coco = COCOData(COCO_JSON_PATH, RESNET_FEATURES_PATH)

print("starting caption vectorizer...")
vectorizer = CaptionVectorizer(coco=coco)

print("loading image-to-caption model and weights...")
model = ImageToCaption(in_features = 512, out_features = 200)
if MODEL_WEIGHTS_PATH and Path(MODEL_WEIGHTS_PATH).is_file():
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
else:
    print("no saved weights found, using an untrained model")



# --- smoke-test sizes: bump these up once it's confirmed working ---
MAX_TRAIN_IMAGES = 330000
MAX_VAL_IMAGES = 50000
EPOCHS = 20
# -------------------------------------------------------------------

print("training model...")
train_ids, validation_ids = coco.split_image_ids(train_fraction = 0.8, seed = 0)

train_ids = train_ids[:MAX_TRAIN_IMAGES]
validation_ids = validation_ids[:MAX_VAL_IMAGES]
print(f"training on {len(train_ids)} images ({EPOCHS} epochs)")

train(model=model, train_ids=train_ids, coco=coco, vectorizer=vectorizer, epochs=EPOCHS)

validation_triplets = generate_triplets(coco=coco, allowed_image_ids = validation_ids, num_triplets = len(validation_ids), seed=0)
avg_loss = loss(model=model, coco=coco, vectorizer=vectorizer, triplets=validation_triplets).mean()
print(f"The average validation loss is {avg_loss}")


print("building image database...")
db = ImageDatabase(model, coco)
db.build()

def search_callback(query: str, k: int):
    return db.query(caption_text = query, vectorizer=vectorizer, k=k)


print("launching frontend...")
app=create_frontend(search_function = search_callback)
app.launch(share=False)