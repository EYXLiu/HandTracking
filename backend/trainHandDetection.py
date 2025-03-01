from model import VisionCNN
from dataset import HandmarkDatabase
from display import DisplayImg

import cv2
from dataclasses import dataclass
import torch

from dataloader import DataLoader

@dataclass
class VisionConfig:
    input_layers: int = 3
    filter_1: int = 32
    filter_2: int = 64
    filter_3: int = 128
    
    linear_flatten: int = 128 * 28 * 28
    linear_layer: int = 512
    output_layer: int = 42
    
import time
import os
import random

model = VisionCNN(VisionConfig)
if os.path.exists('model.pth'):
    print("exists")
    state_dict = torch.load('model.pth', weights_only=True)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
model.train()
model = torch.compile(model)

train_dataset = HandmarkDatabase("./hand_keypoint_dataset_26k/images/train", "./hand_keypoint_dataset_26k/coco_annotation/train")
test_dataset = HandmarkDatabase("./hand_keypoint_dataset_26k/images/val", "./hand_keypoint_dataset_26k/coco_annotation/val")

trr = random.randint(0, len(train_dataset))
tsr = random.randint(0, len(test_dataset))

train_loader = DataLoader(train_dataset, trr)
test_loader = DataLoader(test_dataset, tsr)

m = model.to("cpu")
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

max_steps = 2000
val_steps = 20

for i in range(max_steps):
    t0 = time.time()
    last_step = i == max_steps - 1
    if (i != 0 and i % 100 == 0) or last_step:
        s = time.time()
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0
            for _ in range(val_steps):
                x, y, p = test_loader.next_batch()
                pred, loss = model(x, y)
                loss = loss / val_steps
                val_loss_accum += loss.detach()
        
        print(f"Validation Loss at step {i}: {val_loss_accum:.6f} | time: {(time.time() - s) * 1000:2f}ms")
    
    model.train()
    optimizer.zero_grad()
    
    x, y, p = train_loader.next_batch()
    pred, loss = model(x, y)
    loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    t1 = time.time()
    
    dt = (t1 - t0) * 1000
    
    if (i % 50 == 0): 
        print(f"step: {i} | loss: {loss.item():.6f} | norm: {norm:.4f} | time: {dt:.2f}ms")
    
torch.save(model.state_dict(), 'model.pth')


x, y, path = test_loader.next_batch()
pred, loss = model(x, y)
print(f"loss: {loss.item():.6f}")
img = cv2.imread(path)
img = DisplayImg(img, pred)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()