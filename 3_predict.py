import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import rasterio


class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=2, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = [self.add_weight(name=f'tp_{i}', initializer='zeros') for i in range(num_classes)]
        self.fp = [self.add_weight(name=f'fp_{i}', initializer='zeros') for i in range(num_classes)]
        self.fn = [self.add_weight(name=f'fn_{i}', initializer='zeros') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_cls = tf.argmax(y_pred, axis=-1)
        y_true_cls = tf.argmax(y_true, axis=-1)
        for i in range(self.num_classes):
            pred_i = tf.cast(tf.equal(y_pred_cls, i), tf.float32)
            true_i = tf.cast(tf.equal(y_true_cls, i), tf.float32)
            self.tp[i].assign_add(tf.reduce_sum(pred_i * true_i))
            self.fp[i].assign_add(tf.reduce_sum(pred_i * (1 - true_i)))
            self.fn[i].assign_add(tf.reduce_sum((1 - pred_i) * true_i))

    def result(self):
        f1_list = []
        for i in range(self.num_classes):
            precision = self.tp[i] / (self.tp[i] + self.fp[i] + 1e-7)
            recall    = self.tp[i] / (self.tp[i] + self.fn[i] + 1e-7)
            f1_list.append(2 * precision * recall / (precision + recall + 1e-7))
        return tf.reduce_mean(tf.stack(f1_list))

    def reset_state(self):
        for i in range(self.num_classes):
            self.tp[i].assign(0.)
            self.fp[i].assign(0.)
            self.fn[i].assign(0.)
from rasterio import features
from tqdm import tqdm
import geopandas as gpd
import os
from shapely.geometry import shape
from shapely.ops import unary_union
import yaml

# Carrega o arquivo de configuração
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configurações gerais
contextos  = config["contextos"]
model_dir  = config["model_dir"]

# Configurações de predict
predict_cfg     = config["predict"]
batch_size      = predict_cfg["batch_size"]
crs_            = predict_cfg["crs"]
teste           = predict_cfg["teste"]
buffer_dist     = predict_cfg["buffer_dist"]
shapefile_limit = predict_cfg["shapefile_limit"]
output_base     = predict_cfg["output_dir"]
raster_paths    = predict_cfg["raster_paths"]
model_name      = predict_cfg["model_name"]
imagens         = list(raster_paths.keys())

os.makedirs(output_base, exist_ok=True)

if teste:
    print("\n" + "=" * 80)
    print("MODO DE TESTE ATIVO: Serão processados apenas 2 talhões aleatórios.")
    print("=" * 80 + "\n")


def load_image_data(img_type):
    with rasterio.open(raster_paths[img_type]) as src:
        arr = src.read([1, 2, 3, 4, 5, 6, 7])  # Blue, Green, Red, RedEdge, NIR, SWIR1, SWIR2
        transform = src.transform
        crs = src.crs

    arr = np.moveaxis(arr, 0, -1)
    arr_norm = arr.astype(np.float32) / 10000.0

    blue    = arr_norm[:, :, 0]
    green   = arr_norm[:, :, 1]
    red     = arr_norm[:, :, 2]
    rededge = arr_norm[:, :, 3]
    nir     = arr_norm[:, :, 4]
    swir1   = arr_norm[:, :, 5]
    swir2   = arr_norm[:, :, 6]

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
    arr = np.concatenate((arr_norm, evi, savi, nbr1, nbr2, ndre), axis=-1)

    return arr, transform, crs


def pad_image(img, half_context):
    return cv2.copyMakeBorder(
        img, half_context, half_context, half_context, half_context,
        cv2.BORDER_REPLICATE
    )


def predict_geometry(geom, image, transform, model, attr_dict, half_context, desc=""):
    from numpy.lib.stride_tricks import sliding_window_view

    expanded_geom = geom.buffer(buffer_dist)
    mask = features.geometry_mask(
        [expanded_geom], transform=transform, invert=True,
        out_shape=(image.shape[0], image.shape[1])
    )

    padded_img = pad_image(image, half_context)
    prediction_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    context_size = half_context * 2
    n_channels = image.shape[2]

    # view (zero-copy) de todos os patches: shape (H, W, context, context, channels)
    patches = sliding_window_view(padded_img, (context_size, context_size, n_channels))[:, :, 0]

    rows, cols = np.where(mask)
    n_valid = len(rows)

    preds_flat = np.empty(n_valid, dtype=np.uint8)
    for i in tqdm(range(0, n_valid, batch_size), desc=desc, leave=False, unit="batch"):
        br = rows[i:i + batch_size]
        bc = cols[i:i + batch_size]
        preds = model.predict(patches[br, bc], verbose=0)
        preds_flat[i:i + batch_size] = np.argmax(preds, axis=1)

    prediction_map[rows, cols] = preds_flat

    mask_class_1 = (prediction_map == 1).astype(np.uint8)
    shapes = []
    for geom_raster, val in rasterio.features.shapes(mask_class_1, transform=transform):
        if val == 1:
            poly = shape(geom_raster)
            shapes.append(poly)

    if shapes:
        union_poly = unary_union(shapes)
        inter = union_poly.intersection(geom)
        if not inter.is_empty:
            return [{**attr_dict, "geometry": inter}]
    return []


# Carregar shapefile de limites
gdf_limites = gpd.read_file(shapefile_limit)

if teste:
    np.random.seed(config["random_seed"])
    talhoes_amostrados = np.random.choice(gdf_limites.index, size=2, replace=False)
    print(f"Talhões selecionados para teste: {talhoes_amostrados}")
    with open("talhoes_teste.txt", "w") as f:
        for i in talhoes_amostrados:
            f.write(f"{i}\n")

# Loop para cada tipo de imagem
for img_type in imagens:
    print(f"\nCarregando imagem: {img_type}")
    image_data, transform, crs = load_image_data(img_type)

    for context_size in contextos:
        half_context = context_size // 2
        print(f"\n{'='*60}")
        print(f"Contexto {context_size}x{context_size} | imagem: {img_type}")
        print(f"{'='*60}")

        model_path = os.path.join(model_dir, model_name, f"best_model_{context_size}_{model_name}.keras")
        if not os.path.exists(model_path):
            print(f"Modelo não encontrado: {model_path}. Pulando.")
            continue
        model = load_model(model_path, custom_objects={'MacroF1Score': MacroF1Score})

        if teste:
            indices = list(talhoes_amostrados)
            geometrias = ((i, gdf_limites.loc[i]) for i in indices)
        else:
            indices = list(gdf_limites.index)
            geometrias = gdf_limites.iterrows()

        total_talhoes = len(indices)
        all_polygons = []

        for pos, (idx, row) in enumerate(tqdm(geometrias, total=total_talhoes,
                                              desc=f"[ctx{context_size}] talhões", unit="talhão"), start=1):
            desc = f"  talhão {idx} ({pos}/{total_talhoes})"
            attr_dict = row.drop("geometry").to_dict()
            polygons  = predict_geometry(row.geometry, image_data, transform, model, attr_dict,
                                         half_context, desc=desc)
            all_polygons.extend(polygons)

        if all_polygons:
            gdf_out = gpd.GeoDataFrame(all_polygons)
            gdf_out.set_crs(crs, inplace=True)
            gdf_out = gdf_out.to_crs(crs_)
            out_path = os.path.join(output_base, f"class_1_{img_type}_ctx{context_size}.shp")
            gdf_out.to_file(out_path)
            print(f"Shapefile salvo: {out_path}")
        else:
            print(f"Nenhum polígono classe 1 encontrado para {img_type} ctx{context_size}.")
