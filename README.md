# CNN Chats vs Chiens

## Objectif

Ce projet compare deux approches pour classifier des images de chats et de chiens :

- un CNN entraîné à partir de zéro avec au moins 3 blocs convolutionnels;
- un modèle en apprentissage par transfert basé sur ResNet18 pré-entraîné.

Les deux approches utilisent la normalisation par lot, la désactivation aléatoire, l'augmentation de données, un découpage entraînement/validation, le suivi des métriques et la sauvegarde du meilleur point de sauvegarde.

## Environnement

Installation locale :

```bash
pip install -r requirements.txt
```

Sur Google Colab, sélectionner un environnement GPU puis exécuter `notebook.ipynb`.

## Données

Le notebook télécharge automatiquement le jeu de données si le dossier `Cat_Dog_data/` est absent.

Téléchargement manuel possible :

```bash
wget -O Cat_Dog_data.zip https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
unzip Cat_Dog_data.zip
```

Structure attendue :

```text
Cat_Dog_data/
├─ train/
│  ├─ Cat/
│  └─ Dog/
└─ test/
   ├─ Cat/
   └─ Dog/
```

Les données, points de sauvegarde et journaux sont ignorés par Git.

## Entraînement

Ouvrir `notebook.ipynb` puis exécuter les cellules dans l’ordre.

Expériences lancées :

| Expérience | Modèle | Optimiseur | Taux d'apprentissage | Époques | Régularisation |
|---|---|---:|---:|---:|---|
| `scratch_adam` | CNN à partir de zéro | Adam | 1e-3 | 8 | normalisation par lot + désactivation aléatoire 0.5 |
| `scratch_sgd` | CNN à partir de zéro | SGD avec momentum | 1e-2 | 8 | normalisation par lot + désactivation aléatoire 0.5 |
| `transfer_adam` | ResNet18 gelé + tête adaptée | Adam | 1e-3 | 5 | normalisation par lot + désactivation aléatoire 0.4 |
| `transfer_sgd` | ResNet18 gelé + tête adaptée | SGD avec momentum | 1e-2 | 5 | normalisation par lot + désactivation aléatoire 0.4 |

Un planificateur `StepLR(step_size=3, gamma=0.5)` est appliqué dans chaque expérience.

## Évaluation et sauvegardes

Chaque expérience sauvegarde son meilleur modèle dans `checkpoints/`.

Le notebook recharge automatiquement le meilleur point de sauvegarde selon l'exactitude en validation, puis calcule les métriques finales sur le jeu de test :

- perte;
- exactitude;
- précision macro;
- rappel macro;
- matrice de confusion;
- exemples d'images mal classées.

## Résultats

À compléter après exécution du notebook sur GPU. L'environnement local utilisé pour préparer ce dépôt ne contient pas PyTorch, donc les métriques n'ont pas été simulées.

| Expérience | Meilleure époque | Exactitude validation | Précision validation | Rappel validation |
|---|---:|---:|---:|---:|
| `scratch_adam` | à remplir | à remplir | à remplir | à remplir |
| `scratch_sgd` | à remplir | à remplir | à remplir | à remplir |
| `transfer_adam` | à remplir | à remplir | à remplir | à remplir |
| `transfer_sgd` | à remplir | à remplir | à remplir | à remplir |

## Analyse attendue

Après exécution, comparer les courbes générées dans le notebook. L'apprentissage par transfert devrait converger plus vite que le CNN entraîné à partir de zéro, car ResNet18 dispose déjà de filtres visuels génériques appris sur ImageNet. Le CNN entraîné à partir de zéro peut apprendre correctement, mais il demande généralement plus d'époques et présente plus de risque de surapprentissage.

Comparer aussi Adam et SGD : Adam converge souvent plus vite au début, tandis que SGD avec momentum peut produire une généralisation compétitive si le taux d'apprentissage est bien choisi. Utiliser la matrice de confusion et les erreurs typiques pour commenter les confusions entre chats et chiens.

## Limites et pistes d'amélioration

- Augmenter le nombre d'époques pour le CNN entraîné à partir de zéro.
- Tester un ajustement fin partiel de ResNet18 au lieu de geler toutes les couches.
- Tester EfficientNet ou MobileNet pour comparer précision et coût de calcul.
- Ajouter TensorBoard ou Weights & Biases pour journaliser les courbes.
