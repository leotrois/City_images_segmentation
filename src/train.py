from models import Pix2Pix
from torch.utils.data import DataLoader, random_split
import torch
from torchvision.datasets import Cityscapes
import torchvision
import wandb

BATCH_SIZE = 55
IMG_SIZE = (512,512)
EPOCHS = 100
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))


run = wandb.init(
    project="City_images_segmentation",
    # Sauvegarde des hyperparamètres
    config={
        "epochs": EPOCHS,
    })

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(IMG_SIZE),
        torchvision.transforms.ToTensor()
    ]
    
)
data = Cityscapes("./data",split = "train", mode = "fine", target_type = "color", transform = transform, target_transform = transform)
data_test = Cityscapes("./data",split = "train", mode = "fine", target_type = "color", transform = transform, target_transform = transform)






x_train = DataLoader(data, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(data_test, batch_size = 1, shuffle = True)
for batch in x_test:
    batch_test = batch
    break

if __name__ == "__main__":
    model = Pix2Pix().to(device)
    model.train(EPOCHS, x_train,batch_test, device)
    torch.save(model.state_dict(), "Weights_trained.pt")
