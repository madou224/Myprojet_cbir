import sys
import os
import streamlit as st
import numpy as np
from PIL import Image

# Ajouter le répertoire parent de FeatureExtraction au chemin de recherche de Python
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
st.write(f"Chemin ajouté: {base_dir}")

try:
    from FeatureExtraction.utils import calculate_distance
    from FeatureExtraction.extract_features import extract_glcm_features, extract_bit_features
    st.write("Modules importés avec succès.")
except ImportError as e:
    st.write(f"Erreur d'importation: {e}")

# Charger les caractéristiques pré-calculées
try:
    st.write("Chargement des caractéristiques...")
    glcm_features_path = os.path.join(base_dir, 'FeatureExtraction', 'glcm_features.npy')
    bit_features_path = os.path.join(base_dir, 'FeatureExtraction', 'bit_features.npy')
    glcm_features = np.load(glcm_features_path)
    bit_features = np.load(bit_features_path)
    st.write("Caractéristiques chargées.")
except Exception as e:
    st.write(f"Erreur de chargement des caractéristiques: {e}")

# Interface Streamlit
st.title('Recherche d\'Images Basée sur le Contenu')
st.write("Interface chargée.")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.write("Image téléversée.")
    image = Image.open(uploaded_file)
    st.image(image, caption='Image Téléversée.', use_column_width=True)
    
    # Sélection des paramètres
    descriptor = st.selectbox('Choisir le descripteur', ('GLCM', 'BiT'))
    distance_metric = st.selectbox('Choisir la mesure de distance', ('Euclidienne', 'Manhattan', 'Chebyshev', 'Canberra'))
    num_results = st.slider('Nombre d\'images similaires à afficher', 1, 20, 5)
    
    if st.button('Rechercher'):
        st.write("Recherche lancée.")
        try:
            # Extraction des caractéristiques de l'image téléversée
            if descriptor == 'GLCM':
                feature = extract_glcm_features(image)
                dataset_features = glcm_features
            else:
                feature = extract_bit_features(image)
                dataset_features = bit_features

            # Vérifier les caractéristiques extraites
            st.write(f"Caractéristiques extraites de l'image téléversée: {feature}")

            # Calcul des distances
            distances = calculate_distance(feature, dataset_features, distance_metric)
            st.write(f"Distances calculées: {distances}")

            # Chemin du dossier contenant les images
            image_folder = os.path.join(base_dir, 'datasets')
            st.write(f"Chemin du dossier d'images: {image_folder}")

            # Vérifier que le dossier existe
            if not os.path.exists(image_folder):
                st.write(f"Erreur: Le dossier {image_folder} n'existe pas.")
            else:
                # Obtenez la liste des noms de fichiers dans le dossier images
                image_files = []
                for subdir, _, files in os.walk(image_folder):
                    for file in files:
                        if file.endswith('.jpg') or file.endswith('.png'):
                            image_files.append(os.path.join(subdir, file))
                st.write(f"Noms de fichiers d'image: {len(image_files)} images trouvées")

                # Vérifier que le nombre de résultats demandés ne dépasse pas le nombre d'images disponibles
                if num_results > len(image_files):
                    num_results = len(image_files)

                # Affichage des résultats
                sorted_indices = np.argsort(distances)[:num_results]
                st.write(f"Indices triés: {sorted_indices}")
                for idx in sorted_indices:
                    result_image_path = image_files[idx]
                    result_image = Image.open(result_image_path)
                    st.image(result_image, caption=f'Similarité: {distances[idx]:.4f}', use_column_width=True)
        except Exception as e:
            st.write(f"Erreur pendant la recherche: {e}")
else:
    st.write("Aucune image téléversée.")
