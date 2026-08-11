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

from bands_indices import carregar_canais

# Carregar arquivo YAML
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


# Definir contextos e tipos de imagem
contextos = config["contextos"]
imagens = config["imagens"]
bandas_config = config["bandas"]
escala_reflectancia = config.get("escala_reflectancia", 10000.0)
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

# Função para replicar bordas
def add_border_with_replication(src_image, context_size):
    half_context = context_size // 2
    return cv2.copyMakeBorder(
        src_image,
        half_context, half_context, half_context, half_context,
        cv2.BORDER_REPLICATE
    )

# Loop por tipo de imagem
for img_type in imagens:
    print(f"\nPROCESSANDO IMAGEM: {img_type}")

    area = f"{img_type}.tif"
    if not verificar_crs_compatibilidade(area, *[shp[0] for shp in shapefile_classes.values()]):
        print(f"CRS incompatível para a imagem {area}")
        sys.exit()

    # Bandas + índices calculados uma única vez por imagem (independem do context_size)
    imagem_completa, nomes_canais = carregar_canais(area, bandas_config, escala_reflectancia)
    print(f"Canais utilizados ({len(nomes_canais)}): {', '.join(nomes_canais)}")

    with rasterio.open(area) as src_ref:
        for context_size in contextos:
            print(100 * "=")
            print(f"Extraindo dados para context_size={context_size}")
            half_context = context_size // 2

            X = []
            Y = []

            print(f"Iniciando extração para imagem {img_type}")

            padded_image = add_border_with_replication(imagem_completa, context_size)

            pontos_pertencentes = 0
            excluidos = {classe: 0 for classe in shapefile_classes.keys()}

            # Iterar sobre cada classe
            for classe_nome, (shapefile, classe_label) in shapefile_classes.items():
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
                        X.append(context_bands)
                        Y.append(classe_label)
                        pontos_pertencentes += 1
                    else:
                        excluidos[classe_nome] += 1

            print(f"Pontos pertencentes extraídos: {pontos_pertencentes}")

            # Converter listas para arrays
            X = np.array(X)
            Y = np.array(Y)
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
