import numpy as np
import matplotlib.pyplot as plt
import sys
import rasterio
import cv2
from rasterio import features
from tensorflow.keras.utils import to_categorical
import geopandas as gpd
import os
import yaml

# Carregar arquivo YAML
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
    
    
# Definir contextos e tipos de imagem
contextos = config["contextos"]
imagens = config["imagens"]
points_dir = [config["points_dir"]["floresta"], config["points_dir"]["nao_floresta"]]
np.random.seed(config["random_seed"])

# Dicionário de shapefiles com suas respectivas classes
shapefile_classes = {
    "floresta": (gpd.read_file(points_dir[0]), 1),
    "nao_floresta": (gpd.read_file(points_dir[1]), 0),
}

# Função para verificar a compatibilidade de CRS
def verificar_crs_compatibilidade(raster_path, *shapefiles):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    for shapefile in shapefiles:
        if shapefile.crs != raster_crs:
            print(f"CRS incompatível: O shapefile está em {shapefile.crs}, mas o raster está em {raster_crs}.")
            return False
    print("Todos os CRS são compatíveis.")
    return True

# Loop por tipo de imagem
for img_type in imagens:
    print(f"\nPROCESSANDO IMAGEM: {img_type}")
    

    area = f"{img_type}.tif"
    if not verificar_crs_compatibilidade(area, *[shp[0] for shp in shapefile_classes.values()]):
        print(f"CRS incompatível para a imagem {area}")
        sys.exit()

    for context_size in contextos:
        print(100 * "=")
        print(f"Extraindo dados para context_size={context_size}")
        half_context = context_size // 2

        X = []
        Y = []

        print(f"Iniciando extração para imagem {img_type}")

        # Função para replicar bordas
        def add_border_with_replication(src_image, context_size):
            return cv2.copyMakeBorder(
                src_image,
                half_context, half_context, half_context, half_context,
                cv2.BORDER_REPLICATE
            )

        with rasterio.open(area) as src:
            original_image = src.read([1, 2, 3, 4, 5, 6, 7])  # Blue, Green, Red, RedEdge, NIR, SWIR1, SWIR2
            original_image = np.moveaxis(original_image, 0, -1)
            padded_image = add_border_with_replication(original_image, context_size)
            src_ref = src

        pontos_pertencentes = 0
        excluidos = {classe: 0 for classe in shapefile_classes.keys()}

        # Iterar sobre cada classe
        for classe_nome, (shapefile, classe_label) in shapefile_classes.items():
            with rasterio.open(area) as src_dummy:
                for idx, point in shapefile.iterrows():
                    coords = (point.geometry.x, point.geometry.y)
                    row, col = src_ref.index(*coords)
                    row += half_context
                    col += half_context
                    context_bands = padded_image[
                                  row - half_context:row + half_context,
                                  col - half_context:col + half_context,
                                  :
                                  ]

                    if context_bands.shape[:2] == (context_size, context_size):
                        # Normalizar bandas para 0-1 (Sentinel-2 escala 0–10000)
                        bands   = context_bands.astype(np.float32) / 10000.0
                        blue    = bands[:, :, 0]
                        green   = bands[:, :, 1]
                        red     = bands[:, :, 2]
                        rededge = bands[:, :, 3]
                        nir     = bands[:, :, 4]
                        swir1   = bands[:, :, 5]
                        swir2   = bands[:, :, 6]

                        evi  = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-5)
                        savi = 1.5 * (nir - red) / (nir + red + 0.5 + 1e-5)
                        nbr1 = (nir - swir1) / (nir + swir1 + 1e-5)
                        nbr2 = (nir - swir2) / (nir + swir2 + 1e-5)
                        ndre = (nir - rededge) / (nir + rededge + 1e-5)

                        evi  = np.expand_dims(evi,  axis=-1)
                        savi = np.expand_dims(savi, axis=-1)
                        nbr1 = np.expand_dims(nbr1, axis=-1)
                        nbr2 = np.expand_dims(nbr2, axis=-1)
                        ndre = np.expand_dims(ndre, axis=-1)

                        # 7 bandas (norm.) + EVI, SAVI, NBR1, NBR2, NDRE = 12 canais
                        context_combined = np.concatenate((bands, evi, savi, nbr1, nbr2, ndre), axis=-1)

                        X.append(context_combined)
                        Y.append(classe_label)
                        pontos_pertencentes += 1
                    else:
                        excluidos[classe_nome] += 1

        print(f"Pontos pertencentes extraídos: {pontos_pertencentes}")

        # Converter listas para arrays e normalizar
        X = np.array(X)
        Y = np.array(Y)
        # Bandas e índices já normalizados durante a extração
        transformed_X = X

        # One-hot encoding
        num_classes = len(shapefile_classes)
        Y = to_categorical(Y, num_classes=num_classes)

        # Embaralhar
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        transformed_X = transformed_X[indices]
        Y = Y[indices]

        # Exibir info
        print(f"Número de amostras em X = {transformed_X.shape}")
        print(f"Número de amostras em Y = {Y.shape}")

        # Criar pasta de saída
        os.makedirs("context_data", exist_ok=True)

        # Salvar
        img_name = os.path.splitext(os.path.basename(img_type))[0]
        np.save(fr'context_data/X_context_rgb_{context_size}_{img_name}.npy', transformed_X)
        np.save(fr'context_data/Y_labels_{context_size}_{img_name}.npy', Y)

        forest_count     = int(Y[:, 1].sum())
        non_forest_count = int(Y[:, 0].sum())
        print(f"  Floresta:     {forest_count} ({100*forest_count/len(Y):.1f}%)")
        print(f"  Não-floresta: {non_forest_count} ({100*non_forest_count/len(Y):.1f}%)")
        print(f"Arquivos salvos para {img_name}.")
