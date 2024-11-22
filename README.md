# Implémentation de Pix2Pix

Ce projet est une implémentation du modèle Pix2Pix, un réseau de neurones pour la traduction d'images par paires, tel que décrit dans le papier "Image-to-Image Translation with Conditional Adversarial Networks" par Isola et al. Nous allons utiliser ce modèle pour colorer des images de comics.
Sur l'image suivante, la première colonne représente l'image d'origine, la deuxième l'image en noir et blanc et la troisième la prédiction du modèle.

![Results](images/results.jpg)

## Structure du projet

- `train.py` : Script pour entraîner le modèle.
- `infer.py` : Script pour générer des images avec le modèle entraîné.
- `model.py` : Définition de l'architecture du modèle Pix2Pix.
- `utils.py` : Fonctions utilitaires pour le traitement des données et l'entraînement.
- `classes.py` : Définition des structures de dataloader corresspondant à notre jeu de donnée 

### Détails de l'architecture Pix2Pix

Le modèle Pix2Pix se compose de deux parties principales : un générateur et un discriminateur.

- **Générateur** : Le générateur prend une image d'entrée et génère une image de sortie. Il utilise une architecture de type U-Net, qui est un réseau de neurones convolutif avec des connexions de saut entre les couches correspondantes de l'encodeur et du décodeur. Cela permet de préserver les détails de l'image d'entrée tout en générant l'image de sortie.

- **Discriminateur** : Le discriminateur évalue la qualité des images générées par le générateur. Il utilise une architecture de type PatchGAN, qui divise l'image en petits patchs et évalue chaque patch individuellement. Cela permet de se concentrer sur les détails locaux de l'image.

### Flux de travail

1. **Prétraitement des données** : Les images d'entrée et de sortie sont normalisées et redimensionnées pour correspondre aux dimensions attendues par le modèle.
2. **Entraînement** : Le générateur et le discriminateur sont entraînés conjointement. Le générateur essaie de tromper le discriminateur en générant des images réalistes, tandis que le discriminateur essaie de distinguer les images générées des images réelles.
3. **Évaluation** : Après l'entraînement, le modèle peut être utilisé pour générer des images à partir de nouvelles images d'entrée.

### Traitement des données
Pour que le jeu de donnée soit compatible avec les fonction de création du jeu de donnée, le jeu de donnée doit avoir l'arborescence suivante :

![Data Tree](forme_jeu_donnee_avant_traitement.PNG)

Sinon, vous pouvez utiliser un jeu de donnée qui a directement la forme suivante :
![Data Tree](forme_jeu_donnee.PNG)

### Jeu de donnée
Par la suite, nous utilisons le jeu de donnée de comics des années 50 issu du papier ["The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives"](https://arxiv.org/abs/1611.05118). Vous pouvez accéder au jeu de donnée via ce [lien](https://obj.umiacs.umd.edu/comics/index.html) (on utilise la version original panel images).
### Résultats
Les résultats suivants sont obtenus pour un entrainement de XX epochs sur le modèle.
## Référence

- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004)

## Auteur

- Léo Soudre

