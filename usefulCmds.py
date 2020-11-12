# Chargement des packages utiles

import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


# Charger le fichier CSV
DATASET_PATH = r'./dataset/imgs/'
LABEL_PATH = r'./dataset/labels/labels.csv'
dataset_df = pd.read_csv(LABEL_PATH)

# We add another column to the labels dataset to identify image path
dataset_df['image_path'] = dataset_df.apply( lambda row: (DATASET_PATH + row["id"] ), axis=1)

# Chargement des images
feature_values = np.array([plt.imread(img).reshape(40000,) for img in dataset_df['image_path'].values.tolist()])

# Récupération des labels
label_names = dataset_df['seafloor']
label_names_unique = label_names.unique()

#  transformation des labels selon différents codages
# indices
le = preprocessing.LabelEncoder()
le.fit(label_names_unique)
label_indices = le.transform(label_names_unique)

# one-hot-encoding
label_ohe = pd.get_dummies(label_names.reset_index(drop=True)).values
