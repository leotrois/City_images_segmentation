import os
import shutil
from tqdm import tqdm
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

PATH = "/raid/home/automatants/soudre_leo/Mini_projet_Automatants/Dataset"
queue = Queue()

def remove_BW(PATH = "./Dataset"):
    '''
    Cette fonction a pour but de supprimer les images en noir et blanc.
    '''
    for vol in tqdm(os.listdir(PATH)):
        try:
            shutil.rmtree(os.path.join(PATH, vol, "pages_BW"))
        except:
            pass
        
def rgb_to_bw(image_path, bw_image_path):
    image = cv.imread(image_path)
    
    if image is not None:
        bw_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imwrite(bw_image_path, bw_image)
        
def process_image(vol, image_name):
    image_path = os.path.join(PATH, vol, "Colour", image_name)
    bw_image_path = os.path.join(PATH, vol, "pages_BW", "BW" + image_name)
    rgb_to_bw(image_path, bw_image_path)
    queue.put((image_path, bw_image_path))

def process_volume(vol):
    vol_path = os.path.join(PATH, vol, "Colour")
    bw_dir = os.path.join(PATH, vol, "pages_BW")
    
    try:
        os.mkdir(bw_dir)
    except FileExistsError:
        print("Warning: directory already exists.")
        shutil.rmtree(bw_dir)
        os.mkdir(bw_dir)
    
    with ThreadPoolExecutor() as executor:
        for image_name in os.listdir(vol_path):
            executor.submit(process_image, vol, image_name)

def initialize_dataset():
    remove_BW()
    dataset = open('dataset.csv', "w+")
    dataset.write("Image,BW_Image,Image_Test\n")
    
    volumes = os.listdir(PATH)
    for vol in tqdm(volumes):
        process_volume(vol)
    
    while not queue.empty():
        image_path, bw_image_path = queue.get()
        dataset.write(f"{image_path},{bw_image_path}\n")
    
    dataset.close()

if __name__ == "__main__":
    initialize dataset()