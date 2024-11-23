from models import Pix2Pix
from torchvision.io import read_image
from torchvision import transforms
from torch import permute
from torchvision.utils import save_image
import torch
from torchvision.datasets import Cityscapes
import torchvision
from torch.utils.data import DataLoader


PRETRAINED = False
RANDOM_IMAGE = True
IMG_SIZE = (512, 512)


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
    if RANDOM_IMAGE == False:
        image = read_image('./image_test.jpg')
        image = transforms.Resize(IMG_SIZE)(image)
        image = (image) / 255
        image = image.unsqueeze(0).to(device)
        image = model.forward(image)
        image = torch.squeeze(image)
        save_image(image,"./result.jpg")
    else:
        transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(IMG_SIZE),
            torchvision.transforms.ToTensor()
        ])
        data = Cityscapes("./data",split = "test", mode = "fine", target_type = "color", transform = transform, target_transform = transform)
        x_test = DataLoader(data, batch_size = 1, shuffle = True)

        
        for batch in x_test:
            image = batch[0][0]
            break
        image = image.unsqueeze(0).to(device)
        image = model.forward(image)
        image = torch.squeeze(image)
        save_image(image,"./result.png")
    
    