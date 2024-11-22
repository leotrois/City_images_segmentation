from models import Pix2Pix
from torchvision.io import read_image
from torchvision import transforms
from torch import permute
from torchvision.utils import save_image
import torch

PRETRAINED = True
IMG_SIZE = (512, 512)

import os 
os.chdir(r"C:\Users\Léo\Desktop\Mini_projet_pour_git\Colorisation-de-comics")
if PRETRAINED:
    PATH = "weights_pre_trained.pt"
else:
    PATH = "Weights_trained.pt"
    
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    
if __name__ == "__main__":
    test = torch.load
    model = Pix2Pix()
    model.load_state_dict(torch.load(PATH, weights_only=True, map_location=torch.device(device) ))
    
    model.to(device)
    
    image = read_image('./image_test.jpg')
    image = transforms.Resize(IMG_SIZE)(image)
    image = (image - 127.5) / 127.5
    image = image.unsqueeze(0).to(device)
    image = model.forward(image)
    image = torch.squeeze(image)
    save_image(image,"./result.jpg")
    
    