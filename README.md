# ViT_Rollout_MRI
Projet de MLOps Machine learning end to end allant du dataset ( interpretation, pre processing, analyse) en passant par le développement jusqu'au déployement avec une notion d'explicabilité nécessaire dans notre cas d'étude à savoir l'imagerie médicale. Notre dataset est composé d'images médicales cérébrales à classifier grâce au machine learning. Deux algorithmes basés sur la méthode transformer sont proposés : transformer + heatmap, transformer + attention rollout.

*Cf la fin du readme pour avoir plus d'explication sur les transformers et l'explicabilité.*

# Dataset

Source : https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

## Analyse du dataset

![image](https://user-images.githubusercontent.com/82390666/219058109-df1c1b61-66d8-44d3-acd1-ecd75eb4567b.png)
![image](https://user-images.githubusercontent.com/82390666/219058278-e607f910-d798-4da0-9ec6-aa75442bba70.png)
![image](https://user-images.githubusercontent.com/82390666/219058357-a9a45296-90e7-4dc9-9990-c8b39c906824.png)
![image](https://user-images.githubusercontent.com/82390666/219058461-a870f507-e60c-4e63-89c9-ce83fada5085.png)
![image](https://user-images.githubusercontent.com/82390666/219058541-defb8947-dd5f-4298-b36c-2006146b4dd3.png)
![image](https://user-images.githubusercontent.com/82390666/219058702-62746e3e-6f59-4875-b507-934377e874db.png)

**Pour la suite du projet on ne gardera que deux classes : no_tumor et glioma_tumor, de manière à réaliser une classification binaire.**

# Vision Rollout

source : https://keras.io/examples/vision/probing_vits/

Permet d'afficher les zone de concentration du ML sur les images étudiées. 
L'algorithme "Transformer_rollout" permet d'utiliser le model vit_b16_patch16_224. On importe le dataset, on preprocess les images en changeant leur taille à 224*224 puis on charge le modèle. 

"
preprocessed_image = preprocess_image(image, "original_vit")
predictions, attention_score_dict = vit_base_i21k_patch16_224.predict(preprocessed_image)
attn_rollout_result = attention_rollout_map(pic, attention_score_dict, model_type="original_vit")
"

Sont les étapes à suivre pour obtenir la map final.
Une partie classification en compilant et entraînant le modèle est ensuite détaillé mais présente des problèmes de RAM avec collab gratuit.

# Heatmap

# Transformers
Les algorithmes de Transformer avec déploiement d'attention ou cartes thermiques sont des techniques pour visualiser et comprendre comment un modèle de vision Transformer fait des prédictions. Ils peuvent être utiles pour améliorer la transparence et la compréhension des modèles de vision, ce qui peut être important dans des domaines tels que la médecine où les décisions du modèle peuvent avoir des conséquences importantes pour les patients. Les algorithmes de Transformer avec déploiement d'attention permettent de visualiser les régions de l'image qui ont été les plus influentes pour une prédiction donnée. Cela peut aider à comprendre comment le modèle interagit avec les différentes parties de l'image pour faire des prédictions. Les algorithmes de cartes thermiques fonctionnent de manière similaire, en utilisant une représentation visuelle pour montrer les régions de l'image qui ont été les plus importantes pour une prédiction. Dans le cas d'un projet de classification de tumeurs cérébrales à l'aide d'un modèle de Vision Transformer, ces algorithmes pourraient être utiles pour comprendre comment le modèle fait des prédictions pour différents types de tumeurs et comment il utilise les informations de l'image pour faire des prédictions. Cela peut aider à identifier des domaines où le modèle peut être amélioré ou des sources potentielles de biais, ce qui peut améliorer la qualité et la fiabilité des résultats.

# Explicabilité
L'explicabilité en intelligence artificielle (AI) est la capacité de comprendre et de décrire les décisions et les prédictions d'un modèle de AI. Dans le cas d'un projet de classification de tumeurs cérébrales à l'aide d'un modèle de Vision Transformer, l'explicabilité est importante car les décisions du modèle peuvent avoir des conséquences importantes pour les patients. Il est donc important de comprendre comment le modèle fait des prédictions et de s'assurer que les décisions sont fiables et valides.

L'utilisation de l'explicabilité en AI peut aider à garantir la qualité et la fiabilité des résultats du modèle de Vision Transformer en permettant de comprendre les décisions du modèle et de les valider. Par exemple, en utilisant les algorithmes de déploiement d'attention ou de cartes thermiques décrits précédemment, il est possible de visualiser les régions de l'image qui ont été les plus influentes pour une prédiction donnée, ce qui peut aider à comprendre comment le modèle utilise les informations de l'image pour faire des prédictions.

En outre, l'explicabilité peut également aider à détecter des sources potentielles de biais dans les données ou le modèle, ce qui peut améliorer la qualité et la fiabilité des résultats du modèle. En utilisant l'explicabilité en AI dans ce projet, il est possible d'améliorer la transparence et la compréhension des décisions du modèle de Vision Transformer, ce qui peut aider à garantir que les résultats sont fiables et valides pour la classification de tumeurs cérébrales.

# Choix des outils
Keras et scikit-learn (Sklearn) sont tous deux des bibliothèques populaires pour le développement de modèles d'apprentissage automatique en Python. Toutefois, ils ont des objectifs différents et des fonctionnalités spécifiques qui les rendent plus appropriés pour certaines tâches.

Keras est une bibliothèque d'apprentissage profond qui se concentre sur la simplification du développement de réseaux de neurones complexes. Il fournit une interface haut niveau pour construire, former et évaluer des modèles d'apprentissage profond en utilisant des couches prédéfinies et des algorithmes optimiseurs. Cela en fait un choix populaire pour les projets de reconnaissance d'images, de traitement du langage naturel et de génération de contenu.

D'un autre côté, Sklearn est une bibliothèque pour les algorithmes d'apprentissage supervisé et non supervisé. Il offre un large éventail d'algorithmes, tels que les régressions, les arbres de décision, les k-means, etc. avec une interface cohérente et simple à utiliser. Sklearn est souvent utilisé pour les projets de classification, de régression et de clustering.

En résumé, Keras peut être un choix plus approprié pour les projets d'apprentissage profond tels que la classification

