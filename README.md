# Implémentation de Pix2Pix

Une partie du code de ce projet est tiré d'un projet précédent de [Colorisation de comics](github.com/leotrois/Colorisation-de-comics)


La suite du readme n'est pas à jour...



Ce projet est une implémentation du modèle Pix2Pix, un réseau de neurones pour la traduction d'images par paires, tel que décrit dans le papier "Image-to-Image Translation with Conditional Adversarial Networks" par Isola et al. Nous allons utiliser ce modèle pour segmenter des photos prises par des voitures. Pour cela nous utiliserons le [Cityscapes Dataset](cityscapes-dataset.com/)



## Structure du projet

- `train.py` : Script pour entraîner le modèle.
- `infer.py` : Script pour générer des images avec le modèle entraîné.
- `model.py` : Définition de l'architecture du modèle Pix2Pix.

### Détails de l'architecture Pix2Pix

Le modèle Pix2Pix se compose de deux parties principales : un générateur et un discriminateur.

- **Générateur** : Le générateur prend une image d'entrée et génère une image de sortie. Il utilise une architecture de type U-Net, qui est un réseau de neurones convolutif avec des connexions de saut entre les couches correspondantes de l'encodeur et du décodeur. Cela permet de préserver les détails de l'image d'entrée tout en générant l'image de sortie.

- **Discriminateur** : Le discriminateur évalue la qualité des images générées par le générateur. Il utilise une architecture de type PatchGAN, qui divise l'image en petits patchs et évalue chaque patch individuellement. Cela permet de se concentrer sur les détails locaux de l'image.

### Flux de travail

1. **Prétraitement des données** : Les images d'entrée et de sortie sont normalisées et redimensionnées pour correspondre aux dimensions attendues par le modèle.
2. **Entraînement** : Le générateur et le discriminateur sont entraînés conjointement. Le générateur essaie de tromper le discriminateur en générant des images réalistes, tandis que le discriminateur essaie de distinguer les images générées des images réelles.
3. **Évaluation** : Après l'entraînement, le modèle peut être utilisé pour segmenter des images à partir de nouvelles images d'entrée.


### Résultats
Les résultats suivants sont obtenus pour un entrainement de 10 epochs sur le modèle.



## Auteur

- Léo Soudre

