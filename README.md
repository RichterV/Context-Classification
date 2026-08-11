# Classificação por Contexto

*Exemplo de delimitação de áreas com florestas plantadas a partir de imagens do Sentinel L2A*

![Exemplo de delimitação de áreas com florestas a partir de imagens do Sentinel L2A](exemplo.gif) ![Exemplo de delimitação de áreas com florestas a partir de imagens do Sentinel L2A](exemplo2.gif)

Pipeline de aprendizado profundo para classificação de pixels de imagens de satélite usando UNet++. A partir de pontos de amostra criados em qualquer software GIS, o pipeline treina um modelo que classifica cada pixel de um raster `.tif` e gera um shapefile com os polígonos de cada classe detectada.

## Como funciona

O modelo recebe uma janela (`contexto × contexto` pixels) centrada em cada pixel e decide sua classe. Usar uma janela ao invés do pixel isolado permite que o modelo capture padrões de textura, forma e vizinhança, o que aumenta significativamente a precisão.

**Entrada:** raster de qualquer sensor (satélite, drone, etc.) + pontos de amostra por classe em shapefiles + shapefile com os limites da área a classificar.

**Saída:** shapefile vetorial com os polígonos de cada classe dentro dos limites informados.

**Canais de entrada por pixel:** as bandas brutas são as que o usuário informar em `bandas` no `config.yml` (nome lógico → número da banda no raster), normalizadas para 0–1. A partir dessas bandas, o pipeline calcula automaticamente todo índice espectral cujas bandas dependentes estejam disponíveis — nenhum código precisa ser alterado ao trocar de sensor, só o `config.yml`. O número final de canais (bandas + índices calculáveis) é detectado dinamicamente pelo treino.

Índices reconhecidos (implementados em `bands_indices.py`):

| Índice | Bandas exigidas | Descrição |
|--------|------------------|-----------|
| NDVI | NIR, Red | Normalized Difference Vegetation Index |
| SAVI | NIR, Red | Soil-Adjusted Vegetation Index |
| EVI | NIR, Red, Blue | Enhanced Vegetation Index |
| NBR1 | NIR, SWIR1 | Normalized Burn Ratio (NIR/SWIR1) |
| NBR2 | NIR, SWIR2 | Normalized Burn Ratio (NIR/SWIR2) |
| NDRE | NIR, RedEdge | Normalized Difference Red Edge |
| MPRI | Green, Red | Modified Photochemical Reflectance Index (funciona em drones RGB, sem NIR) |
| GNDVI | NIR, Green | Green Normalized Difference Vegetation Index |

Um índice só é calculado se **todas** as bandas que ele exige estiverem presentes em `bandas`. Se faltar, por exemplo, SWIR1/SWIR2 (comum em imagens de drone), NBR1/NBR2 são simplesmente omitidos e o pipeline segue apenas com as bandas brutas + índices ainda calculáveis.

---

## Estrutura do projeto

```
context_classification/
├── config.yml                              # Configurações centralizadas de todo o pipeline
├── bands_indices.py                        # Registro de bandas/índices espectrais (usado pela extração e predição)
├── 1_extract_train_data_from_points.py     # Extrai patches de treino dos pontos amostrais
├── 2_train.py                              # Treina o modelo UNet++ por contexto
├── 3_predict.py                            # Prediz sobre o raster e gera shapefile
├── 4_pos_processamento_shps.py             # Pós-processa o shapefile gerado
├── visualize_context_size.py               # Utilitário: visualiza patches aleatórios do raster
├── teste_gpu.py                            # Utilitário: verifica se a GPU está disponível
├── shapefiles/
│   ├── classe_a.shp                        # Pontos de amostra da classe A (ex: floresta)
│   ├── classe_b.shp                        # Pontos de amostra da classe B (ex: não floresta)
│   └── limite_fazenda.shp                  # Polígonos dos talhões/área a classificar
└── raster/
    └── imagem.tif                          # Raster Sentinel-2 de entrada
```

> Pastas `context_data/`, `context_models/`, `logs/`, `predictions_talhoes/` e `pos_processados/` são geradas automaticamente pelo pipeline e devem estar no `.gitignore`.

---

## Pré-requisitos

- Python 3.9+
- TensorFlow 2.x com suporte a GPU (recomendado)
- GDAL/rasterio

```bash
pip install tensorflow rasterio geopandas shapely numpy opencv-python scikit-learn tqdm pyyaml matplotlib
```

Para verificar se a GPU está sendo reconhecida:

```bash
python teste_gpu.py
```

---

## Configuração — `config.yml`

Todos os parâmetros do pipeline estão em um único arquivo:

```yaml
contextos: [64]           # tamanhos de janela (pixels) a treinar/predizer
imagens: ['raster/nome']  # nome(s) do(s) raster(s) de treino (sem extensão .tif)

points_dir:
  classe_a: shapefiles/classe_a.shp   # shapefile de pontos positivos
  classe_b: shapefiles/classe_b.shp   # shapefile de pontos negativos

bandas:                    # nome lógico -> número da banda no raster (1-based)
  blue: 1                  # remova a linha de uma banda que não existir na sua imagem
  green: 2
  red: 3
  rededge: 4
  nir: 5
  swir1: 6
  swir2: 7

escala_reflectancia: 10000 # opcional; padrão 10000 (Sentinel-2). Use 1 se a imagem já vier em 0-1

training:
  epochs: 5000
  batch_size: 32
  patience: 100           # early stopping
  learning_rate: 0.001

predict:
  batch_size: 256
  crs: "EPSG:31982"
  teste: false            # true = processa apenas 2 talhões (verificação rápida)
  buffer_dist: 3
  shapefile_limit: "shapefiles/limite_fazenda.shp"
  model_name: "nome_do_modelo"
  raster_paths:
    predict_raster: "raster/imagem.tif"

pos_processamento:
  input_shapefile: "predictions_talhoes/class_1_predict_raster_ctx64.shp"
  filtrar_poligonos: true
  filtrar_aneis: true
  simplificar_geometria: true
  validar_geometria: true
  min_area_poligonos: 30      # m²
  min_area_aneis: 15          # m²
  tolerancia_simplify: 1      # metros
```

---

## Pipeline — passo a passo

### 1. Preparar dados de amostra

No seu software GIS (QGIS, ArcGIS, etc.), crie um shapefile de pontos para cada classe que deseja classificar — por exemplo, `floresta.shp` e `nao_floresta.shp`. Os pontos devem estar sobre a imagem `.tif` e ter o mesmo CRS do raster.

Configure os caminhos no `config.yml` em `points_dir`.

### 2. Extrair patches de treino

```bash
python 1_extract_train_data_from_points.py
```

Para cada ponto amostral e para cada tamanho de contexto configurado em `contextos`, extrai uma janela `contexto × contexto` pixels centrada no ponto, lê as bandas definidas em `bandas`, calcula os índices espectrais compatíveis com essas bandas (ver tabela acima) e salva os arrays em `context_data/`.

Arquivos gerados:
- `context_data/X_context_rgb_{contexto}_{imagem}.npy` — patches de entrada (N, ctx, ctx, n_canais), onde `n_canais` = nº de bandas informadas + nº de índices calculáveis
- `context_data/Y_labels_{contexto}_{imagem}.npy` — rótulos one-hot (N, 2)

### 3. Treinar o modelo

```bash
python 2_train.py
```

Treina uma UNet++ para cada combinação de `(contexto, imagem)` configurada. O modelo recebe um patch `contexto × contexto × n_canais` (número de canais inferido automaticamente do `.npy` gerado na extração) e classifica o pixel central.

**Arquitetura:**
- Encoder: 3 blocos Conv2D → MaxPooling com 32/64/128/256 filtros
- Decoder: skip connections densas no estilo UNet++ (nós x01, x11, x21, x02, x12, x03)
- Head: GlobalAveragePooling2D → Dense(64) → Dense(n_classes, softmax)

**Callbacks:** EarlyStopping, ModelCheckpoint (melhor val_f1_score), ReduceLROnPlateau, TensorBoard.

Modelo salvo em: `context_models/{imagem}/best_model_{contexto}_{imagem}.keras`

Para acompanhar o treino:

```powershell
tensorboard --logdir logs\nome_da_imagem
# Acesse: http://localhost:6006
```

### 4. Predizer sobre o raster

```bash
python 3_predict.py
```

Para cada talhão do shapefile de limites:
1. Aplica um buffer de `buffer_dist` pixels na geometria
2. Cria uma view zero-copy de todos os patches via `sliding_window_view`
3. Roda `model.predict` em batches sobre os pixels dentro da máscara
4. Converte o mapa de predição em polígonos via `rasterio.features.shapes`
5. Intersecta os polígonos com a geometria original do talhão

> **Modo teste:** defina `teste: true` no `config.yml` para processar apenas 2 talhões aleatórios e validar rapidamente o resultado antes de rodar a predição completa.

Shapefile de saída: `predictions_talhoes/class_1_{imagem}_ctx{contexto}.shp`

### 5. Pós-processar o shapefile

```bash
python 4_pos_processamento_shps.py
```

Aplica correções ao shapefile de predição (configuradas por flags no `config.yml`):

| Flag | O que faz |
|------|-----------|
| `filtrar_poligonos` | Remove polígonos com área < `min_area_poligonos` m² |
| `filtrar_aneis` | Remove buracos internos com área < `min_area_aneis` m² |
| `simplificar_geometria` | Suaviza bordas "escada" de pixels com tolerância configurável |
| `validar_geometria` | Corrige auto-intersecções e geometrias inválidas via `make_valid` |

O shapefile de entrada nunca é modificado. Resultado salvo em: `pos_processados/{nome}_corrigido.shp`

---

## Utilitários

### Visualizar patches do raster

```python
# visualize_context_size.py — edite as variáveis ao final do arquivo
caminho_arquivo = "raster/imagem.tif"
caminho_shp = "shapefiles/limite_fazenda.shp"
n = 64  # tamanho do contexto a visualizar
```

Exibe patches aleatórios dentro dos limites para inspecionar visualmente o tamanho de janela antes de treinar.

---

## Resultados e escolha do contexto

Experimentos com imagens Sentinel-2 upscaladas para 1m×1m (via S2DR3) mostraram que o tamanho de contexto ideal é **64 pixels (~64m de janela)**:

| Contexto | Accuracy | F1     |
|----------|----------|--------|
| 32       | 0.9537   | 0.9537 |
| 40       | 0.9564   | 0.9564 |
| 56       | 0.9608   | 0.9608 |
| **64**   | **0.9626** | **0.9626** |
| 72       | 0.9559   | 0.9559 |
| 80       | 0.9493   | 0.9493 |

A acurácia cresce monotonicamente de 8 até 64 e começa a degradar a partir de 72, indicando que janelas muito grandes introduzem contexto irrelevante. Para outros sensores ou resoluções, recomenda-se testar a faixa 32–96.

---

## Citação

O uso deste repositório, no todo ou em parte, requer citação obrigatória do autor e da fonte. Use o formato mais adequado ao seu caso — cada bloco abaixo pode ser copiado pelo ícone no canto superior direito.

**Texto simples:**

```
Richter, Vinicius. Context Classification. 2026. Disponível em: https://github.com/RichterV/Context-Classification
```

**APA:**

```
Richter, V. (2026). Context Classification [Software]. GitHub. https://github.com/RichterV/Context-Classification
```

**BibTeX:**

```bibtex
@software{richter2026contextclassification,
  author = {Richter, Vinicius},
  title  = {Context Classification},
  year   = {2026},
  url    = {https://github.com/RichterV/Context-Classification}
}
```
