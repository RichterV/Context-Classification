"""
Cálculo dinâmico de canais de entrada (bandas brutas + índices espectrais).

O usuário informa em `config.yml` um dicionário `bandas` mapeando nomes lógicos
(blue, green, red, rededge, nir, swir1, swir2, ...) para o número da banda no
raster. Cada índice em INDICES declara quais bandas lógicas exige; um índice só
é calculado se todas as bandas necessárias estiverem presentes no dicionário do
usuário. Bandas ausentes simplesmente desativam os índices que dependem delas —
o pipeline continua funcionando apenas com as bandas brutas informadas.
"""
import numpy as np
import rasterio

# nome_do_indice: (bandas_logicas_exigidas, formula)
INDICES = {
    "ndvi":  (["nir", "red"],
              lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 1e-5)),
    "savi":  (["nir", "red"],
              lambda b: 1.5 * (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.5 + 1e-5)),
    "evi":   (["nir", "red", "blue"],
              lambda b: 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 1 + 1e-5)),
    "nbr1":  (["nir", "swir1"],
              lambda b: (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"] + 1e-5)),
    "nbr2":  (["nir", "swir2"],
              lambda b: (b["nir"] - b["swir2"]) / (b["nir"] + b["swir2"] + 1e-5)),
    "ndre":  (["nir", "rededge"],
              lambda b: (b["nir"] - b["rededge"]) / (b["nir"] + b["rededge"] + 1e-5)),
    "mpri":  (["green", "red"],
              lambda b: (b["green"] - b["red"]) / (b["green"] + b["red"] + 1e-5)),
    "gndvi": (["nir", "green"],
              lambda b: (b["nir"] - b["green"]) / (b["nir"] + b["green"] + 1e-5)),
}


def indices_calculaveis(bandas_config):
    """Nomes dos índices cujas bandas exigidas estão todas presentes em `bandas_config`."""
    return [nome for nome, (bandas_necessarias, _) in INDICES.items()
            if all(b in bandas_config for b in bandas_necessarias)]


def carregar_canais(raster_path, bandas_config, escala=10000.0):
    """
    Lê do raster apenas as bandas informadas em `bandas_config`
    (dict nome_logico -> número da banda no raster), normaliza para 0-1
    e concatena os índices espectrais calculáveis com essas bandas.

    Retorna (array HxWxC float32, lista com o nome de cada canal na ordem em que aparece).
    """
    nomes_bandas = list(bandas_config.keys())

    with rasterio.open(raster_path) as src:
        arr = src.read([bandas_config[nome] for nome in nomes_bandas])
    arr = np.moveaxis(arr, 0, -1).astype(np.float32) / escala

    bandas = {nome: arr[:, :, i] for i, nome in enumerate(nomes_bandas)}

    canais = [arr]
    nomes_canais = list(nomes_bandas)
    for nome_indice in indices_calculaveis(bandas_config):
        _, formula = INDICES[nome_indice]
        valor = np.expand_dims(formula(bandas), axis=-1)
        canais.append(valor)
        nomes_canais.append(nome_indice)

    return np.concatenate(canais, axis=-1), nomes_canais
