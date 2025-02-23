import cv2

from model import VisionCNN
from display import DisplayImg

from dataclasses import dataclass
import torchvision.transforms as transforms

import torch

@dataclass
class VisionConfig:
    input_layers: int = 3
    filter_1: int = 32
    filter_2: int = 64
    filter_3: int = 128
    
    linear_flatten: int = 128 * 28 * 28
    linear_layer: int = 512
    output_layer: int = 42

model_path = "model.pth"
model = VisionCNN(VisionConfig)
state_dict = torch.load(model_path, weights_only=True)
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (224, 224))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = transform(image)
        x = model.predict(image)
        img = DisplayImg(img, x)
        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()