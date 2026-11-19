# Création, entraînement, évaluation et exploration de modèles génératifs avec PyTorch sur MNIST

Travaux pratiques, partie 1 : classifieur, autoencodeur et VAE.

Cours de **Mohamed Youssfi**, enseignant-chercheur à l'ENSET, Université Hassan II.

Le notebook principal est `Modeles_generatifs_pytorch.ipynb`. Il contient le cours du professeur : MNIST, classifieur, autoencodeur, VAE, puis les sections GAN et diffusion. Seule la première partie est exécutée pour ce TP. Elle utilise le noyau **Python 3.11 (TP MNIST)** et, par défaut, recharge les poids de `models/`.

Les autres notebooks sont conservés dans `archives/`. Le notebook principal ne les importe pas et n'en a pas besoin pour s'exécuter. Le PDF du cours reste à la racine du projet.

## 1. Objectifs

Le même jeu MNIST sert à enchaîner quatre questions du cours : reconnaître, représenter, reconstruire, puis générer.

- Le **classifieur** estime une approximation de \(P(y \mid x)\). Il transforme une image en dix logits et choisit la classe de score maximal.
- L'**autoencodeur** apprend un code latent ponctuel \(z\) de dimension 2, puis reconstruit l'image. Les étiquettes n'entrent pas dans la perte.
- Le **VAE** remplace ce point par une gaussienne diagonale \(q(z \mid x)\). La divergence de Kullback-Leibler rapproche cette loi de \(\mathcal{N}(0, I)\). On peut ensuite décoder un code tiré dans la loi de référence.

Les hyperparamètres, les architectures et les formules sont ceux du notebook et du support du professeur. L'adaptation locale concerne le périphérique de calcul : MPS sur Apple Silicon, CPU si MPS est inutilisable. CUDA n'est pas sélectionné.

## 2. MNIST

MNIST rassemble des chiffres manuscrits de 0 à 9, en niveaux de gris, de taille \(28 \times 28\), soit 784 pixels. Le cours en utilise 60 000 pour l'entraînement et 10 000 pour le test. Ces deux effectifs ont été vérifiés après le téléchargement par `torchvision.datasets.MNIST`.

`transforms.ToTensor()` produit un tenseur `float32` de forme `[1, 28, 28]` et ramène les niveaux 0–255 dans l'intervalle **[0, 1]**. Aucune normalisation par moyenne et écart-type n'est ajoutée : la sigmoïde des décodeurs et l'entropie croisée binaire du VAE s'interprètent dans cet intervalle.

Mesures relevées sur le premier lot d'entraînement, de taille 128 :

| Objet | Forme | Dtype |
| --- | --- | --- |
| Image unitaire | `[1, 28, 28]` | `float32` |
| Lot d'images | `[128, 1, 28, 28]` | `float32` |
| Lot d'étiquettes | `[128]` | `int64` |

Les pixels de ce lot vont de 0 à 1. La première image d'entraînement est un 5.

Effectifs par classe, de 0 à 9 :

| Découpage | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Entraînement | 5 923 | 6 742 | 5 958 | 6 131 | 5 842 | 5 421 | 5 918 | 6 265 | 5 851 | 5 949 |
| Test | 980 | 1 135 | 1 032 | 1 010 | 982 | 892 | 958 | 1 028 | 974 | 1 009 |

Les dix classes sont présentes des deux côtés. Le jeu de test sert à l'évaluation et aux figures. Il ne sert à aucune mise à jour de poids.

## 3. Environnement technique

Exécution réalisée le 19 novembre 2026.

| Élément | Valeur vérifiée |
| --- | --- |
| Système | macOS, Apple Silicon (arm64) |
| Shell | zsh |
| UV | 0.13.9 |
| Python du projet | 3.11.18, dans `.venv` |
| PyTorch | 2.15.0 |
| Torchvision | 0.29.2 |
| Torcheval | 0.0.7 |
| Matplotlib | 3.11.2 |
| NumPy | 2.4.6 |
| Jupyter | 1.1.1, avec ipykernel 7.4.0 |
| Périphérique | `mps` |
| MPS construit / disponible | oui / oui |
| CUDA | non disponible, et non utilisé |
| Graine | 42 |
| Taille de lot | 128 |

`PYTORCH_ENABLE_MPS_FALLBACK=1` est posé au démarrage du notebook pour laisser un opérateur non implémenté sur MPS retomber sur le CPU. Aucun repli de ce type n'a été nécessaire pour les trois modèles : l'entraînement, l'évaluation et `torcheval` ont tourné sur MPS.

La graine fixe l'initialisation et les tirages contrôlés par un `torch.Generator`. Sur MPS, certains noyaux restent non déterministes. Une seconde exécution peut donc différer légèrement, à architecture et hyperparamètres identiques.

## 4. Installation et exécution avec UV

Le dépôt est déjà initialisé (`uv init` a été fait une seule fois). Ne pas relancer `uv init`.

Depuis la racine du projet, celle qui contient `pyproject.toml` :

```bash
uv venv --python 3.11
uv sync
```

`uv sync` installe le verrou `uv.lock` : PyTorch, Torchvision, Torcheval, Matplotlib, NumPy, ipykernel et Jupyter.

Pour enregistrer le noyau dans l'environnement du projet, puis ouvrir le notebook principal :

```bash
uv run python -m ipykernel install --sys-prefix --name=tp-mnist --display-name="Python 3.11 (TP MNIST)"
uv run jupyter lab Modeles_generatifs_pytorch.ipynb
```

Dans Cursor, choisir le noyau **Python 3.11 (TP MNIST)**. Son interpréteur est `.venv/bin/python`.

`FORCE_RETRAIN` vaut `False` dans la première cellule de code. Les poids compatibles de `models/` sont rechargés, et les historiques de `results/` sont seulement lus. Passer la constante à `True`, puis exécuter le notebook depuis le début, recrée chaque réseau et l'entraîne depuis zéro pour la session : 3, 5 et 20 epochs. Cette branche n'écrit pas dans `models/` ni dans `results/`.

## 5. Structure du projet

```text
Modeles_Generatifs_MNIST/
├── Modeles_generatifs_pytorch.ipynb          # notebook principal
├── Part 2- IA Générative Multi-Modale - Modèles génératfs V4.pdf
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── archives/
│   ├── Modeles_generatifs_pytorch.ipynb.bak
│   ├── Modeles_generatifs_pytorch_avec_explications.ipynb
│   └── TP_MNIST_Modeles_Generatifs.ipynb
├── data/                 # MNIST téléchargé, ignoré par git
├── models/
│   ├── classifier.pth
│   ├── autoencoder.pth
│   └── vae.pth
├── results/              # figures, historiques et metrics.json
└── src/                  # paquet créé par uv init
```

Le nom du PDF conserve la graphie du fichier d'origine. `.gitignore` écarte `.venv`, les caches Python, les checkpoints Jupyter et le contenu de `data/`, tout en conservant `data/.gitkeep`. Les poids et les figures restent dans le projet.

## 6. Les trois modèles

### Classifieur

`Flatten`, puis `Linear(784, 128)`, ReLU, `Linear(128, 10)`.

`Flatten` change `[B, 1, 28, 28]` en `[B, 784]` et n'a aucun paramètre. La dernière couche produit des logits. `CrossEntropyLoss` applique elle-même le log-softmax : le réseau ne se termine pas par une softmax. Les étiquettes sont indispensables. Le modèle compte **101 770** paramètres.

Il apprend une décision parmi dix classes. Il ne décode pas d'image.

### Autoencodeur

Dimension latente 2.

- Encodeur : `Flatten`, `784 → 128 → 32 → 2`, ReLU après les deux premières couches linéaires.
- Décodeur : `2 → 32 → 128 → 784`, ReLU, puis sigmoïde. Le vecteur est remis en forme `[batch, 1, 28, 28]`.

La dimension 2 est celle du cours : elle permet de dessiner le plan latent, au prix d'une compression sévère. Le modèle compte **210 130** paramètres. L'apprentissage ignore les étiquettes.

### VAE

Dimension latente 2.

- Encodeur : `Flatten`, `Linear(784, 256)`, ReLU, `Linear(256, 64)`, ReLU.
- Deux têtes : `fc_mu` et `fc_logvar`, de 64 vers 2.
- Reparamétrisation : \(z = \mu + \exp(0{,}5 \log \sigma^2) \odot \epsilon\), avec \(\epsilon \sim \mathcal{N}(0, I)\).
- Décodeur : `2 → 64 → 256 → 784`, ReLU, sigmoïde, puis remise en forme `[batch, 1, 28, 28]`.

Le décodeur produit, pour chaque pixel, une valeur dans [0, 1], lue comme une probabilité d'activation et affichée en niveau de gris. Le modèle compte **435 988** paramètres.

## 7. Hyperparamètres et fonctions de perte

| Réglage | Classifieur | Autoencodeur | VAE |
| --- | --- | --- | --- |
| Epochs | 3 | 5 | 20 |
| Optimiseur | Adam | Adam | Adam |
| Taux d'apprentissage | 0,001 | 0,001 | 0,001 |
| Taille de lot | 128 | 128 | 128 |
| Dimension latente | — | 2 | 2 |
| Perte | entropie croisée | MSE | BCE sommée + \(\beta\) KL |

Les pertes suivent le cours.

\[
\mathcal{L}_{\mathrm{class}} = \mathrm{CrossEntropy}(\hat{y}, y)
\]

La valeur tracée est la moyenne des pertes de lots. Chaque perte de lot est déjà une moyenne sur les images du lot.

\[
\mathcal{L}_{\mathrm{rec}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2
\]

`MSELoss` moyenne tous les pixels du lot. La courbe reprend la moyenne de ces pertes de lots. La MSE de test moyenne, elle, tous les pixels de test, chacun une fois.

\[
D_{\mathrm{KL}} = -\frac{1}{2}\sum\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)
\]

\[
\mathcal{L}_{\mathrm{VAE}} = \mathcal{L}_{\mathrm{rec}} + \beta D_{\mathrm{KL}}, \qquad \beta = 1
\]

`binary_cross_entropy` est appelée avec `reduction="sum"`. La KL est aussi une somme. Les deux termes sont ensuite divisés par le nombre d'images, ce qui donne une perte **par image**. Avec \(\beta = 1\), le critère est celui du VAE classique.

`model.train()` encadre les mises à jour. L'évaluation et les figures passent par `model.eval()` et `torch.no_grad()`. Ces réseaux n'ont ni Dropout ni BatchNorm : `eval()` ne supprime pas le tirage de \(z\). La perte de test du VAE est donc un passage stochastique, lancé après avoir reposé la graine.

## 8. Lancer le notebook dans Cursor

1. Ouvrir le dossier `Modeles_Generatifs_MNIST` comme racine du workspace.
2. Sélectionner l'interpréteur `.venv/bin/python` (Python 3.11.18).
3. Ouvrir `Modeles_generatifs_pytorch.ipynb`.
4. Choisir le noyau **Python 3.11 (TP MNIST)**.
5. Exécuter les cellules dans l'ordre jusqu'à la fin du VAE, avant la partie GAN.

`FORCE_RETRAIN = False` recharge les poids compatibles et affiche l'historique déjà mesuré dans `results/`. Les fichiers `.pth` et les figures existantes ne sont pas remplacés. Pour un entraînement depuis zéro, passer la constante à `True` et relancer depuis la première cellule : les réseaux sont recréés, puis entraînés 3, 5 et 20 epochs, sans remplacer les poids du disque. La partie GAN et la partie diffusion restent dans le notebook ; elles ne sont pas nécessaires pour cette première partie.

## 9. Résultats mesurés

Toutes les valeurs ci-dessous proviennent de l'exécution du 19 novembre 2026, périphérique `mps`, graine 42. Elles sont aussi dans `results/metrics.json` et dans les sorties du notebook. Rien n'a été estimé à la main.

### Classifieur

| Époque | Cross-entropy moyenne des lots |
| ---: | ---: |
| 1 | 0,4107 |
| 2 | 0,1921 |
| 3 | 0,1392 |

Accuracy de test, `torcheval.metrics.MulticlassAccuracy` sur MPS, confirmée par un décompte manuel des prédictions : **96,31 %**. Durée d'entraînement : **5,3 s**.

Rappel par classe, de 0 à 9 : 98,67 % ; 99,21 % ; 96,80 % ; 97,52 % ; 95,42 % ; 96,08 % ; 96,66 % ; 95,43 % ; 92,71 % ; 94,15 %. Le chiffre 8 est le moins bien rappelé, le chiffre 1 le mieux rappelé.

Sur la planche de dix images de test, neuf prédictions sont exactes. Le 5 de cette planche est lu comme un 6. Dans la matrice complète, les confusions les plus nombreuses sont un 8 prédit 3 (23 images), un 4 prédit 9 (21), un 5 prédit 3 (18), un 9 prédit 3 (16) et un 9 prédit 4 (16).

### Autoencodeur

| Époque | MSE moyenne des lots |
| ---: | ---: |
| 1 | 0,0655 |
| 2 | 0,0517 |
| 3 | 0,0477 |
| 4 | 0,0457 |
| 5 | 0,0447 |

MSE globale sur les 10 000 images de test : **0,044472**. Durée d'entraînement : **11,8 s**.

Les reconstructions de test sont floues. Les traits simples, notamment les 1 et le 0 de la planche, restent lisibles. Plusieurs 4, 5, 7 et 9 perdent le détail qui les distingue : c'est le coût attendu d'un code à deux nombres.

L'espace latent du test s'étale bien au-delà de \([-3, 3]\). Les 0 occupent plutôt le haut du nuage et les 1 le bas droit. Le centre mélange fortement les autres classes. Aucune perte n'a demandé à ces coordonnées de suivre une gaussienne.

Le décodage du code moyen de chaque chiffre, calculé sur le test sans mise à jour des poids, donne un prototype net pour 0 et pour 1. Les autres prototypes restent ambigus. Parmi dix codes qui ne sont pas des moyennes de données, plusieurs points proches de l'origine produisent encore une forme de chiffre, parce qu'ils tombent près des codes appris. Le point \((8{,}5,\ -12)\) cité dans le support dégénère en un simple trait. L'autoencodeur peut décoder un vecteur ; il n'a pas appris de loi \(p(z)\).

### VAE

Perte par image. La reconstruction et la KL sont les deux termes de la somme, avec \(\beta = 1\).

| Époque | Total | Reconstruction | KL |
| ---: | ---: | ---: | ---: |
| 1 | 194,49 | 190,34 | 4,14 |
| 20 | 141,47 | 135,01 | 6,46 |

La perte totale et la reconstruction diminuent à chaque époque. La KL augmente régulièrement de 4,14 à 6,46 : le posterior ne s'est pas effondré sur le prior, et il ne s'est pas non plus affranchi de la régularisation. À l'époque 20, la KL représente environ 4,6 % de la perte totale. Durée d'entraînement : **59,9 s**.

Sur le test, un passage stochastique donne :

| Terme | Valeur par image |
| --- | ---: |
| Perte totale | 142,64 |
| Reconstruction | 136,21 |
| KL | 6,43 |

Les reconstructions des mêmes dix images de test sont, elles aussi, adoucies. La forme globale est souvent conservée, y compris pour le 7 de la planche, tandis que les confusions 4/9 et la forme du 5 restent difficiles en deux dimensions.

Les vingt images tirées avec \(z \sim \mathcal{N}(0, I)\) ressemblent à des chiffres, avec des traits épais et quelques formes ambiguës. La grille \(15 \times 15\) sur \([-3, 3]^2\) montre des régions contiguës, notamment pour 0, 1, 3, 5, 6, 7, 8 et 9, et des transitions progressives. Certains chiffres n'occupent qu'une bande étroite.

Le notebook principal montre cette continuité par la grille du cours. Le fichier `results/vae_interpolation.png`, issu de la session de mesure, suit en plus le segment entre le premier 0 de test, \(\mu = [-0{,}665,\ 2{,}010]\), et le premier 1, \(\mu = [-1{,}040,\ -2{,}275]\). Ce segment traverse des formes intermédiaires avant d'atteindre le 1.

## 10. Figures générées

Les fichiers ci-dessous sont dans `results/`, produits pendant la session de mesure du 19 novembre 2026. Le notebook principal réaffiche, à chaque exécution de la première partie, l'échantillon MNIST, les reconstructions, les espaces latents, le code moyen d'un chiffre, les vingt échantillons du VAE et la grille \(15 \times 15\). Il ne réécrit pas ces fichiers.

| Fichier | Contenu |
| --- | --- |
| `mnist_echantillon.png` | Dix images du premier lot d'entraînement et leurs étiquettes |
| `mnist_distribution_classes.png` | Effectifs des dix classes, entraînement et test |
| `classifier_loss.png` | Cross-entropy moyenne des lots, 3 époques |
| `classifier_predictions.png` | Dix images de test, classe réelle et classe prédite |
| `classifier_matrice_confusion.png` | Matrice de confusion sur les 10 000 images de test |
| `autoencoder_loss.png` | MSE moyenne des lots, 5 époques |
| `autoencoder_reconstructions.png` | Dix originales de test et leurs reconstructions |
| `autoencoder_espace_latent.png` | Codes \(z\) du test, couleur = étiquette |
| `autoencoder_exploration_generation.png` | Codes moyens des dix chiffres, puis dix codes libres |
| `vae_loss.png` | Perte totale, reconstruction et KL, par image |
| `vae_reconstructions.png` | Dix originales de test et reconstructions échantillonnées |
| `vae_espace_latent.png` | Moyennes \(\mu\) du test, couleur = étiquette |
| `vae_echantillons.png` | Vingt images issues de \(z \sim \mathcal{N}(0, I)\) |
| `vae_grille_latente.png` | Grille \(15 \times 15\) de \(-3\) à \(3\) |
| `vae_interpolation.png` | Dix décodages entre le \(\mu\) d'un 0 et celui d'un 1 |

Les historiques chiffrés sont dans `results/classifier_history.json`, `results/autoencoder_history.json`, `results/vae_history.json` et `results/metrics.json`.

## 11. Comparaison

|  | Classifieur | Autoencodeur | VAE |
| --- | --- | --- | --- |
| Question | Quelle est la classe ? | Comment reconstruire \(x\) ? | Comment reconstruire et tirer une image nouvelle ? |
| Étiquettes dans la perte | utilisées | ignorées | ignorées |
| Sortie | 10 logits | image dans [0, 1] | image dans [0, 1] |
| Loi imposée à \(z\) | aucune | aucune | \(\mathcal{N}(0, I)\), via la KL |
| Paramètres | 101 770 | 210 130 | 435 988 |
| Durée mesurée | 5,3 s | 11,8 s | 59,9 s |
| Lecture sur le test | accuracy 96,31 % | MSE 0,044472 | perte 142,64 ; reconstruction 136,21 ; KL 6,43 |

Ces trois nombres ne se rangent pas sur une même échelle.

- L'**accuracy** est une proportion de décisions exactes.
- La **MSE** moyenne un carré d'écart par pixel, pour des intensités dans [0, 1].
- La **perte du VAE** additionne, pour chaque image, une entropie croisée binaire sur 784 pixels et une KL sur deux coordonnées. Sa valeur est grande parce qu'elle est une somme par image, pas une moyenne par pixel.

Le classifieur discrimine. L'autoencodeur reconstruit et compresse. Le VAE reconstruit et propose un mécanisme de génération : tirer \(z\) dans la loi de référence, puis décoder. La grille et l'interpolation montrent ce que ce décodeur a rendu continu. Les échantillons restent flous, et certains sont ambigus.

## 12. Limitations et difficultés rencontrées

- La dimension latente 2, imposée par le cours pour la visualisation, limite la fidélité. Les reconstructions et les échantillons sont flous. Une MSE ou une entropie croisée plus basse ne dirait pas, à elle seule, que le chiffre est net.
- L'autoencodeur organise partiellement les classes, sans loi de référence. Décoder un code arbitraire peut encore donner un chiffre s'il tombe près des données, et dégénère pour un point éloigné comme \((8{,}5,\ -12)\). Ce comportement confirme l'absence de prior ; il ne faut pas le lire comme un générateur fiable.
- Le VAE de cette partie est un perceptron, entraîné 20 époques, avec une vraisemblance pixel à pixel. La KL mesurée reste du même ordre entre l'entraînement (6,46) et le test (6,43) : pas d'effondrement du posterior sur cette exécution. Le compromis laisse néanmoins des régions où plusieurs chiffres se recouvrent, ce que l'interpolation de 0 vers 1 traverse.
- MPS a fonctionné, y compris pour `MulticlassAccuracy`. Torcheval 0.0.7 émet toutefois un `FutureWarning` : il passe encore par `torch.jit.script`, signalé comme déprécié dans PyTorch 2.15. La métrique a été calculée, et elle coïncide avec le décompte manuel.
- MPS n'est pas entièrement déterministe. La graine stabilise l'initialisation et les tirages explicites ; elle ne promet pas une copie bit à bit d'une exécution à l'autre.
- Dans `results/vae_interpolation.png`, les titres sont arrondis à une décimale : le pas \(t = 5/9 \approx 0{,}56\) s'affiche 0,6. Les dix images du segment sont présentes. Le notebook principal explore le plan par la grille \(15 \times 15\).
- Le téléchargement de MNIST écrit sa progression sur la sortie d'erreur du notebook. Ce n'est pas un échec : les 60 000 et 10 000 images ont été chargées.
- L'interpréteur `python3` du système est en 3.12. Le projet utilise le Python 3.11.18 du `.venv` créé par `uv venv --python 3.11`.
- Le notebook doit être lancé depuis la racine du projet. Un autre dossier courant interrompt la première cellule.

## 13. Conclusion

Sur des images MNIST ramenées à [0, 1], trois contrats d'apprentissage ont été entraînés puis lus sur le test.

Le classifieur répond à la question discriminative du cours. En 3 époques, sa cross-entropy moyenne passe de 0,4107 à 0,1392 et son accuracy de test atteint 96,31 %. Les erreurs restantes se concentrent sur des chiffres visuellement proches, en particulier 8/3, 4/9 et 5/3.

L'autoencodeur, en 5 époques, abaisse la MSE moyenne des lots de 0,0655 à 0,0447, et la MSE de test vaut 0,044472. Il fournit une carte à deux dimensions et des reconstructions grossières. Le décodage d'un code moyen s'appuie encore sur des images réelles. Le décodage d'un code éloigné rappelle qu'aucune loi \(p(z)\) n'a été apprise.

Le VAE, en 20 époques, fait passer la perte par image de 194,49 à 141,47. Sur le test, elle vaut 142,64, dont 136,21 de reconstruction et 6,43 de KL. L'espace des moyennes est resserré autour de l'origine, les classes y forment des régions, et le décodeur produit des chiffres nouveaux à partir de \(\mathcal{N}(0, I)\). La continuité se voit sur la grille et sur le segment qui relie un 0 à un 1. La netteté reste celle d'un modèle à deux variables latentes et d'une perte pixel à pixel.

La première partie s'arrête avant les GAN et la diffusion, qui restent dans le notebook principal pour la suite du cours. Elle fixe le vocabulaire utile : une perte accordée à la question, un test tenu à l'écart de l'entraînement, des formes de tenseurs cohérentes, et une lecture explicite de l'espace latent.

## 14. Documents du professeur

- Youssfi, M. Notebook principal `Modeles_generatifs_pytorch.ipynb`, *Modèles génératifs avec PyTorch*.
- Youssfi, M. Notebook commenté, archivé dans `archives/Modeles_generatifs_pytorch_avec_explications.ipynb`.
- Youssfi, M. Copie du notebook avant adaptation, `archives/Modeles_generatifs_pytorch.ipynb.bak`.
- Youssfi, M. Support `Part 2- IA Générative Multi-Modale - Modèles génératfs V4.pdf`, *IA Générative Multi-Modale — Introduction à l'IA générative, modèles génératifs*.

Références scientifiques du jeu de données et du modèle variationnel repris dans ce TP :

- LeCun, Y., Cortes, C. et Burges, C. *The MNIST database of handwritten digits*.
- Kingma, D. P. et Welling, M. *Auto-Encoding Variational Bayes*, 2014.
