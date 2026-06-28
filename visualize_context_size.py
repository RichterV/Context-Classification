import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from shapely.geometry import box
import random

def mostrar_porcao_aleatoria_tif(caminho_arquivo, caminho_shp, n, qtd=10, max_tentativas=2000):
    with rasterio.open(caminho_arquivo) as src:
        imagem = src.read([3, 2, 1])
        transform = src.transform
        raster_crs = src.crs

        imagem_np = np.moveaxis(imagem, 0, -1)
        altura, largura = imagem_np.shape[:2]

        if n > altura or n > largura:
            raise ValueError(f"O valor de n ({n}) é maior que as dimensões da imagem ({altura}x{largura})")

    gdf = gpd.read_file(caminho_shp)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    area_valida = gdf.union_all()

    mostradas = 0
    tentativas = 0

    while mostradas < qtd and tentativas < max_tentativas:
        tentativas += 1
        x_ini = random.randint(0, largura - n)
        y_ini = random.randint(0, altura - n)

        # Converte pixels para coordenadas geográficas
        x_min, y_max = transform * (x_ini, y_ini)
        x_max, y_min = transform * (x_ini + n, y_ini + n)
        patch_geom = box(x_min, y_min, x_max, y_max)

        if area_valida.contains(patch_geom):
            porcao = imagem_np[y_ini:y_ini + n, x_ini:x_ini + n].astype(np.float32)
            p2, p98 = np.percentile(porcao, (2, 98))
            porcao = np.clip((porcao - p2) / (p98 - p2 + 1e-6), 0, 1)
            plt.imshow(porcao)
            plt.title(f"Patch {mostradas + 1} — pixel ({x_ini}, {y_ini})")
            plt.axis('off')
            plt.show()
            plt.close()
            mostradas += 1

    print(f"{mostradas}/{qtd} patches exibidos em {tentativas} tentativas.")

# Exemplo de uso
caminho_arquivo = "raster/nome_imagem.tif"
caminho_shp = "shapefiles/limite_fazenda.shp"
n = 40
mostrar_porcao_aleatoria_tif(caminho_arquivo, caminho_shp, n)
