from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg import Connection
from psycopg.sql import Identifier
from psycopg.sql import SQL
from shapely import from_wkb
from shapely.geometry import MultiLineString

from edge_offset.geojson import Feature
from edge_offset.geojson import write_feature_collection
from edge_offset.linework import coerce_multiline_geometry
from edge_offset.linework import merge_multiline_geometries
from edge_offset.offset_linework import offset_polygon_from_classified_polygon
from edge_offset.rings import classify_polygon_from_edge_sets


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    identificatie: str
    underpass_id: int | None
    movable_edges: MultiLineString
    fixed_edges: MultiLineString


def load_edge_records_from_db(
    connection: Connection[Any],
    *,
    edges_table: Identifier,
) -> tuple[EdgeRecord, ...]:
    query = SQL(
        """
        SELECT
            identificatie::text,
            underpass_id,
            edge_type,
            ST_AsBinary(geom) AS edge_wkb
        FROM {edges_table}
        WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom)
        ORDER BY identificatie, underpass_id, edge_type
        """
    ).format(edges_table=edges_table)

    # Group edges by identificatie and underpass_id
    edge_groups: dict[tuple[str, int], dict[str, list[bytes]]] = {}

    with connection.cursor() as cursor:
        cursor.execute(query)
        for row in cursor.fetchall():
            identificatie, underpass_id, edge_type, edge_wkb = row
            key = (str(identificatie), int(underpass_id))

            if key not in edge_groups:
                edge_groups[key] = {"exterior": [], "shared": [], "interior": []}

            if edge_type in edge_groups[key] and edge_wkb is not None:
                edge_groups[key][edge_type].append(edge_wkb)

    # Build EdgeRecord objects
    records: list[EdgeRecord] = []

    for (identificatie, underpass_id), edge_types in edge_groups.items():
        # Convert exterior edges to movable_edges MultiLineString
        exterior_geometries = [
            _load_geometry_from_wkb(wkb) for wkb in edge_types["exterior"]
        ]
        movable_edges = merge_multiline_geometries(*exterior_geometries)

        # Convert shared and interior edges to fixed_edges MultiLineString
        shared_geometries = [
            _load_geometry_from_wkb(wkb) for wkb in edge_types["shared"]
        ]
        interior_geometries = [
            _load_geometry_from_wkb(wkb) for wkb in edge_types["interior"]
        ]
        fixed_edges = merge_multiline_geometries(
            *(shared_geometries + interior_geometries)
        )

        records.append(
            EdgeRecord(
                identificatie=identificatie,
                underpass_id=underpass_id,
                movable_edges=movable_edges,
                fixed_edges=fixed_edges,
            )
        )

    return tuple(records)


def offset_polygon_features_from_db(
    connection: Connection[Any],
    *,
    edges_table: Identifier,
    distance: float,
    tolerance: float = 1e-6,
    strategy: str = "boolean_patch",
) -> list[Feature]:
    records = load_edge_records_from_db(
        connection,
        edges_table=edges_table,
    )

    features: list[Feature] = []
    for record in records:
        # print(f"Processing underpass {record.identificatie} (ID: {record.underpass_id}) with {len(record.movable_edges.geoms)} movable edges and {len(record.fixed_edges.geoms)} fixed edges.")
        classified_polygon = classify_polygon_from_edge_sets(
            movable_edges=record.movable_edges,
            fixed_edges=record.fixed_edges,
            tolerance=tolerance,
        )
        polygon = offset_polygon_from_classified_polygon(
            classified_polygon,
            distance=distance,
            tolerance=tolerance,
            strategy=strategy,
        )
        features.append(
            Feature(
                geometry=polygon,
                properties={
                    "identificatie": record.identificatie,
                    "underpass_id": record.underpass_id,
                    "offset_distance": distance,
                    "strategy": strategy,
                },
            )
        )

    return features


def write_offset_polygons_from_db(
    connection: Connection[Any],
    *,
    edges_table: Identifier,
    distance: float,
    output_path: Path,
    tolerance: float = 1e-6,
    strategy: str = "boolean_patch",
) -> list[Feature]:
    features = offset_polygon_features_from_db(
        connection,
        edges_table=edges_table,
        distance=distance,
        tolerance=tolerance,
        strategy=strategy,
    )
    write_feature_collection(features, path=output_path)
    return features


def _load_multiline_from_wkb(value: bytes | memoryview | None) -> MultiLineString:
    return coerce_multiline_geometry(_load_geometry_from_wkb(value))


def _load_geometry_from_wkb(value: bytes | memoryview | None):
    if value is None:
        return None
    return from_wkb(bytes(value))
