from models import Pix2Pix
from torch.utils.data import DataLoader, random_split
import torch
from torchvision.datasets import Cityscapes
import torchvision
BATCH_SIZE = 1
IMG_SIZE = (512,512)
EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(IMG_SIZE),
        torchvision.transforms.ToTensor()
    ]
    
)
data = Cityscapes("./data",split = "train", mode = "fine", target_type = "color", transform = transform, target_transform = transform)
print(data[0][0].size())
print(data[0][1].size())






x_train = DataLoader(data, batch_size = BATCH_SIZE, shuffle = True)

if __name__ == "__main__":
    model = Pix2Pix().to(device)
    model.train(EPOCHS, x_train, device)
    torch.save(model.state_dict(), "Weights_trained.pt")
