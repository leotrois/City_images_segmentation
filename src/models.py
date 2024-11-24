from torch import nn
import torch
from tqdm import tqdm
import wandb
from torchvision import utils,transforms
from PIL import Image
import torchvision
class down(nn.Module):
    
    def __init__(self,in_channels,out_channels,last = False) -> None:
        super().__init__()
        self.couche1 = nn.Conv2d(in_channels=in_channels, out_channels= out_channels,kernel_size=(3,3),  padding='same')
        self.couche2 = nn.Conv2d(in_channels=out_channels, out_channels= out_channels,kernel_size=(3,3),  padding='same')
        self.maxpool = nn.MaxPool2d(kernel_size =(2,2))
        self.dropout = nn.Dropout(0.3)
        self.last = last
    def forward(self,x):
        x = self.couche1(x)
        x = nn.functional.leaky_relu(x)
        x = self.couche2(x)
        if self.last == False:
            x = nn.functional.leaky_relu(x)
            x = self.maxpool(x)
            x = self.dropout(x)
        else:
            x = nn.functional.sigmoid(x)
            x = self.maxpool(x)

        return x
    


class up(nn.Module):
    def __init__(self,in_channel,out_channel,activation = True) -> None:
        super().__init__()
        self.activation = activation # Pour la dernière couche, on veut des valeurs entre -1 et 1
        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel, kernel_size=(2,2), stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        if activation:
            self.conv2d_1 = nn.Conv2d(in_channels=out_channel*2, out_channels= out_channel,kernel_size=(2,2),  padding='same')
        else : 
            self.conv2d_1 = nn.Conv2d(in_channels=out_channel + 3, out_channels= out_channel,kernel_size=(2,2),  padding='same')
 


        self.conv2d_2 = nn.Conv2d(in_channels=out_channel, out_channels= out_channel,kernel_size=(2,2),  padding='same')

    def forward(self,x, connection):
        x = self.conv_transpose(x)  
        x = self.conv2d_1(torch.cat([x , connection],dim=1))
        x = nn.functional.leaky_relu(x)
        x = self.conv2d_2(x)
        if self.activation :
            x = nn.functional.leaky_relu(x)
        else:
            x = torch.nn.functional.sigmoid(x)
        return x
      
          


    
class Unet_with_cat(nn.Module):
    def __init__(self,nb_features) -> None:
        super().__init__()
        self.down1 = down(3,nb_features)  
        self.down2 = down(nb_features,nb_features*2)  
        self.down3 = down(nb_features * 2, nb_features* 4) 

        self.down4 = down(nb_features* 4, nb_features *8) 

        self.up1 = up(nb_features *8, nb_features*4) 
        self.up2 = up(nb_features *4, nb_features*2) 
        self.up3 = up(nb_features *2, nb_features) 
        self.up4 = up(nb_features, 3,activation=False) 
    
    def forward(self,x):
        x0 = x
        x = self.down1(x) 
        x1 = x
        x = self.down2(x)
        x2 = x
        x = self.down3(x) 
        x3 = x
        x = self.down4(x) 
        x = self.up1(x, x3) 
        x = self.up2(x,x2)
        x = self.up3(x,x1) 
        x = self.up4(x,x0) 
        return x
    
    def loss(self,real_images, fake_images, disc_pred):
        
        l1=torch.nn.L1Loss()(real_images,fake_images)
        
        dupage_discriminateur = torch.nn.BCEWithLogitsLoss()(disc_pred,torch.ones_like(disc_pred))

        loss_gen = dupage_discriminateur + 100 * l1
        return loss_gen , dupage_discriminateur, l1
    
    
class Discriminateur(nn.Module):
    
    #Ne marche probablement pas, je n'ai pas réfléchi aux tailles et nombre de canaux
    def __init__(self, nb_features) -> None:
        super().__init__()
        self.down1 = down(6,nb_features)
        self.down2 = down(nb_features,nb_features*2)
        self.down3 = down(nb_features * 2, nb_features* 4)
        self.down4 = down(nb_features * 4, nb_features* 8)
        self.down5 = down(nb_features * 8, 1,last = True) # On veut une grille de pixels
        # En sortie on n'a pas un nombre car Patch Gan !!! Chaque pixel de l'image 
    def forward(self, x, y):
        x = torch.cat([x,y], dim = 1) # (batch size, 256,256,channels*2)
        x = self.down1(x) 
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return x
    def loss(self,disc_real_output, disc_generated_output, bce_loss):
        
        real_loss = bce_loss(disc_real_output,torch.ones_like(disc_real_output))
        generated_loss = bce_loss(disc_generated_output,torch.zeros_like(disc_generated_output))
        
        total_disc_loss= real_loss + generated_loss
        
        return total_disc_loss
    
class Pix2Pix(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.unet = Unet_with_cat(64)
        self.discriminateur = Discriminateur(32)
        self.optimizer_unet = torch.optim.Adam(self.unet.parameters())
        self.optimizer_discriminateur = torch.optim.Adam(self.discriminateur.parameters())

    def forward(self,x):
        return self.unet.forward(x)
        

    def train_step(self, X_batch, Y_batch):
        # On commence par entrainer le discriminateur
        self.discriminateur.zero_grad()
        
        fake_images = self.unet(X_batch)
        real_images = Y_batch[:,:3,:,:]
        disc_real_output = self.discriminateur(real_images,X_batch)
        disc_fake_output = self.discriminateur(fake_images,X_batch)
        gan_loss = self.discriminateur.loss(disc_real_output, disc_fake_output, torch.nn.BCEWithLogitsLoss())
        gan_loss_value = gan_loss.item()
        gan_loss.backward()
        self.optimizer_discriminateur.step()
        
        # On entraine maintenant le générateur
        self.unet.zero_grad()
        fake_images = self.unet(X_batch)
        disc_fake_output = self.discriminateur(fake_images,X_batch)
        loss_tuple = self.unet.loss(real_images, fake_images, disc_fake_output)
        
        loss_value, dupage_discriminateur, l1    = loss_tuple
        
        
        loss_value.backward()
        self.optimizer_unet.step()
        return loss_value.item(), gan_loss_value , dupage_discriminateur.item(), l1.item()
    
    def train(self, epoch, dataloader,batch_test, device):
        for e in tqdm(range(epoch)):
            progression = tqdm(dataloader, colour="#f0768b")
            
            for i, batch in enumerate(progression):
                loss, gan_loss , dupage_discriminateur, l1= self.train_step(batch[0].to(device), batch[1].to(device))
                progression.set_description(f"Epoch {e+1}/{epoch} | loss_gen: {loss} | loss_gan: {gan_loss}")
                wandb.log({"gen_loss":loss, "d_CE_loss":gan_loss, "L1_loss": l1, "g_CE_loss":dupage_discriminateur })
                
            
            input = transforms.ToPILImage()(torch.squeeze(batch_test[0])[:3,:,:])
            pred = transforms.ToPILImage()(torch.squeeze(self.forward(batch_test[0].to(device)))).convert("RGB")
            
            ground_truth = transforms.ToPILImage()(torch.squeeze(batch_test[1].to(device))).convert("RGB")
            

            input = transforms.ToTensor()(input)
            pred = transforms.ToTensor()(pred)
            ground_truth = transforms.ToTensor()(ground_truth)

            
            image_array = utils.make_grid([input, pred, ground_truth],nrow=3)
            images = wandb.Image(
                image_array, caption = f" Left:Input, Center:Output, Right:Ground truth, epoch {e+1}"
            )
            wandb.log({"examples": images})
