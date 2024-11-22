from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas as pd
import torchvision
from torchvision.io import read_image, ImageReadMode
#On crée un dataset custom à partir des images du dossier 'dataset'


class CustomDataset(Dataset):
    def __init__(self, img_csv, img_size, device ="cuda"):
        
        self.img_csv = pd.read_csv(img_csv) 
        self.transform = torchvision.transforms.Resize(img_size)
        self.device = device
    
    def __len__(self):

        return len(self.img_csv)

    def __getitem__(self, index):
        img_path_bw= self.img_csv.iloc[index,1]
        img_path_couleur =self.img_csv.iloc[index,0]
        
        wb, couleur =read_image(img_path_bw, mode = ImageReadMode.GRAY ), read_image(img_path_couleur)
        wb, couleur = self.transform(wb), self.transform(couleur)

        wb, couleur= (wb -127.5)/127.5, (couleur -127.5)/127.5 # We normalise the images
        
        return wb.to(self.device), couleur.to(self.device)
