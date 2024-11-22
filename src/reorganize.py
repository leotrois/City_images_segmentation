import os
from tqdm import tqdm
os.chdir("C:\\Users\\Léo\\Desktop\\Mini_projet_Automatants")

PATH = "./Dataset"

# cette fonction donne une bonne forme au dataset (normalement pas utile pour les GPA)

for vol in tqdm(os.listdir(PATH)):
    try:
        os.mkdir(os.path.join(PATH ,vol, "Colour"))
    except:
        pass
    for page in os.listdir(os.path.join(PATH, vol)):
        try:
            os.rename(os.path.join(PATH, vol, page), os.path.join(PATH, vol, "Colour", page))
        except:
            pass