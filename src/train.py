from models import Pix2Pix
from torch.utils.data import DataLoader, random_split
from classes import CustomDataset
import torch


BATCH_SIZE = 1
IMG_SIZE = (512,512)
EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

data = CustomDataset("./Dataset", IMG_SIZE, device = device)

train_size = int(0.9 * len(data))
test_size = len(data) - train_size



train_dataset, test_dataset = random_split(data, [train_size, test_size])

x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(train_dataset, 1, shuffle = True)

if __name__ == "__main__":
    model = Pix2Pix().to(device)
    model.train(EPOCHS, x_train)
    torch.save(model.state_dict(), "Weights_trained.pt")
