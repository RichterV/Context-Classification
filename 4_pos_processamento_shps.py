import os
import yaml
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

cfg = config["pos_processamento"]

input_shapefile     = cfg["input_shapefile"]
output_dir          = cfg["output_dir"]
filtrar_poligonos   = cfg["filtrar_poligonos"]
filtrar_aneis       = cfg["filtrar_aneis"]
simplificar_geometria = cfg["simplificar_geometria"]
validar_geometria   = cfg["validar_geometria"]
min_area_poligonos  = cfg["min_area_poligonos"]
min_area_aneis      = cfg["min_area_aneis"]
tolerancia_simplify = cfg["tolerancia_simplify"]

base_name   = os.path.splitext(os.path.basename(input_shapefile))[0]
output_path = os.path.join(output_dir, f"{base_name}_corrigido.shp")


def filter_by_area_and_count(geometry, min_area):
    if isinstance(geometry, MultiPolygon):
        initial_count = len(geometry.geoms)
        filtered_polygons = [p for p in geometry.geoms if p.area >= min_area]
        return MultiPolygon(filtered_polygons) if filtered_polygons else None, initial_count - len(filtered_polygons)
    elif isinstance(geometry, Polygon):
        return (geometry, 0) if geometry.area >= min_area else (None, 1)
    return None, 0


def remove_small_holes_and_count(geometry, min_area_m2):
    if isinstance(geometry, Polygon):
        new_interiors = [ring for ring in geometry.interiors if Polygon(ring).area >= min_area_m2]
        holes_removed = len(geometry.interiors) - len(new_interiors)
        return Polygon(shell=geometry.exterior, holes=new_interiors), holes_removed
    elif isinstance(geometry, MultiPolygon):
        new_polygons, total_removed = [], 0
        for poly in geometry.geoms:
            new_poly, removed = remove_small_holes_and_count(poly, min_area_m2)
            new_polygons.append(new_poly)
            total_removed += removed
        return MultiPolygon(new_polygons), total_removed
    return geometry, 0


gdf = gpd.read_file(input_shapefile)
gdf_filtered = gdf.copy()
polygon_removal_count = 0
holes_removal_count = 0

if filtrar_poligonos:
    gdf_filtered['filtered_geometry'] = gdf_filtered['geometry'].apply(
        lambda geom: filter_by_area_and_count(geom, min_area_poligonos))
    gdf_filtered['filtered_geometry'], gdf_filtered['polygon_removed'] = zip(*gdf_filtered['filtered_geometry'])
    polygon_removal_count = gdf_filtered['polygon_removed'].sum()
    gdf_filtered = gdf_filtered[gdf_filtered['filtered_geometry'].notnull()]
    gdf_filtered = gdf_filtered.set_geometry('filtered_geometry')
    gdf_filtered = gdf_filtered.drop(columns=['geometry', 'polygon_removed'])
    gdf_filtered = gdf_filtered.rename(columns={'filtered_geometry': 'geometry'})

if filtrar_aneis:
    gdf_filtered['geometry'], gdf_filtered['holes_removed'] = zip(
        *gdf_filtered['geometry'].apply(lambda geom: remove_small_holes_and_count(geom, min_area_aneis))
    )
    holes_removal_count = gdf_filtered['holes_removed'].sum()
    gdf_filtered = gdf_filtered.drop(columns=['holes_removed'])

if simplificar_geometria:
    gdf_filtered['geometry'] = gdf_filtered['geometry'].simplify(
        tolerancia_simplify, preserve_topology=True
    )

if validar_geometria:
    gdf_filtered['geometry'] = gdf_filtered['geometry'].apply(make_valid)

gdf_filtered = gdf_filtered.set_geometry('geometry')
gdf_filtered = gdf_filtered.set_crs(gdf.crs, allow_override=True)

os.makedirs(output_dir, exist_ok=True)
gdf_filtered.to_file(output_path)

print(f"Polígonos pequenos removidos: {polygon_removal_count}")
print(f"Anéis pequenos removidos: {holes_removal_count}")
print(f"Shapefile salvo em: {output_path}")
