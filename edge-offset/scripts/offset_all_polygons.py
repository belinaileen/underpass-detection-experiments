from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import environ
from pathlib import Path
import sys
import time
from typing import List

from psycopg import connect
from shapely import from_wkb, to_wkb

from edge_offset.linework import coerce_multiline_geometry, merge_multiline_geometries
from edge_offset.offset_linework import GeometryOffsetError
from edge_offset.offset_linework import InvalidInputPolygonError
from edge_offset.offset_linework import offset_polygon_from_classified_polygon
from edge_offset.postgis import EdgeRecord
from edge_offset.rings import classify_polygon_from_edge_sets


ENV_PATH = Path(".env")
CHUNK_SIZE = 1000


def main() -> int:

    _load_dotenv(ENV_PATH)

    distance_value = environ.get("EDGE_OFFSET_OFFSET_DISTANCE")
    if not distance_value:
        raise ValueError("EDGE_OFFSET_OFFSET_DISTANCE must be set.")

    distance = float(distance_value)
    max_workers = int(environ.get("EDGE_OFFSET_MAX_WORKERS", "4"))

    edges_table = environ.get(
        "EDGE_OFFSET_EDGES_TABLE", "underpasses.edges"
    )
    output_table = environ.get(
        "EDGE_OFFSET_OUTPUT_TABLE", "underpasses.extended_geometries"
    )
    skipped_table = environ.get(
        "EDGE_OFFSET_SKIPPED_TABLE", "underpasses.skipped_underpasses"
    )
    id_column = environ.get("EDGE_OFFSET_ID_COLUMN", "underpass_id")

    # Database connection parameters
    db_params = {
        "host": _require_env("EDGE_OFFSET_DB_HOST"),
        "port": int(_require_env("EDGE_OFFSET_DB_PORT")),
        "dbname": _require_env("EDGE_OFFSET_DB_NAME"),
        "user": _require_env("EDGE_OFFSET_DB_USER"),
        "password": environ.get("EDGE_OFFSET_DB_PASSWORD", ""),
    }

    # Setup database and get chunks
    with connect(**db_params) as conn:
        setup_extended_geometries_table(conn, output_table, id_column)
        setup_skipped_underpasses_table(conn, skipped_table, id_column)

        # Check progress
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(DISTINCT {id_column}) FROM {output_table}")
            already_processed = cursor.fetchone()[0]

            cursor.execute(f"SELECT COUNT(DISTINCT {id_column}) FROM {skipped_table}")
            already_skipped = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT COUNT(DISTINCT {id_column}) FROM {edges_table} WHERE geom IS NOT NULL"
            )
            total_underpasses = cursor.fetchone()[0]

        if already_processed > 0 or already_skipped > 0:
            print(
                f"Found {already_processed}/{total_underpasses} already processed underpasses"
            )
            print(
                f"Found {already_skipped}/{total_underpasses} already skipped underpasses"
            )
            remaining = total_underpasses - already_processed - already_skipped
            print(f"Will process remaining {remaining} underpasses")

        underpass_chunks = get_underpass_chunks(
            conn, edges_table, output_table, skipped_table, id_column
        )

    if not underpass_chunks:
        print("All underpasses have been processed! ✅")
        return 0

    print(f"Found {len(underpass_chunks)} chunks to process")
    print(f"Using {max_workers} parallel workers")

    # Process chunks in parallel
    start_time = time.time()
    completed_chunks = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_chunk = {
            executor.submit(
                process_chunk,
                chunk,
                chunk_num + 1,
                distance,
                edges_table,
                output_table,
                skipped_table,
                id_column,
                db_params,
            ): (
                chunk,
                chunk_num + 1,
            )
            for chunk_num, chunk in enumerate(underpass_chunks)
        }

        # Process completed chunks
        for future in as_completed(future_to_chunk):
            chunk, chunk_num = future_to_chunk[future]
            try:
                result = future.result()
                completed_chunks += 1
                elapsed = time.time() - start_time

                print(
                    f"✅ Chunk {chunk_num}/{len(underpass_chunks)} completed: "
                    f"{result['processed']} underpasses, {result['failed']} failures "
                    f"({elapsed:.1f}s elapsed)"
                )

            except Exception as e:
                print(f"❌ Chunk {chunk_num}/{len(underpass_chunks)} failed: {e}")

    total_time = time.time() - start_time
    print(f"\nProcessing completed successfully in {total_time:.1f} seconds")
    return 0


def setup_skipped_underpasses_table(conn, skipped_table, id_column):
    """Create skipped_underpasses table to track failed processing attempts."""
    if id_column == "identificatie":
        schema_sql = """
            CREATE TABLE IF NOT EXISTS {t} (
                identificatie TEXT NOT NULL,
                skip_reason TEXT,
                skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (identificatie)
            );
            CREATE INDEX IF NOT EXISTS idx_skipped_identificatie ON {t} (identificatie);
        """
    else:
        schema_sql = """
            CREATE TABLE IF NOT EXISTS {t} (
                identificatie TEXT NOT NULL,
                underpass_id INTEGER NOT NULL,
                skip_reason TEXT,
                skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (identificatie, underpass_id)
            );
            CREATE INDEX IF NOT EXISTS idx_skipped_underpass_id ON {t} (underpass_id);
            CREATE INDEX IF NOT EXISTS idx_skipped_identificatie ON {t} (identificatie);
        """
    with conn.cursor() as cursor:
        cursor.execute(schema_sql.format(t=skipped_table))
        conn.commit()
    print("Skipped underpasses table ready")


def setup_extended_geometries_table(conn, output_table, id_column):
    """Create extended_geometries table if it doesn't exist (don't drop if exists)."""
    if id_column == "identificatie":
        schema_sql = """
            CREATE TABLE IF NOT EXISTS {t} (
                identificatie TEXT NOT NULL,
                offset_distance DOUBLE PRECISION,
                geom GEOMETRY(POLYGON, 28992),
                PRIMARY KEY (identificatie)
            );
            CREATE INDEX IF NOT EXISTS idx_extended_geom_spatial ON {t} USING GIST (geom);
        """
    else:
        schema_sql = """
            CREATE TABLE IF NOT EXISTS {t} (
                identificatie TEXT NOT NULL,
                underpass_id INTEGER NOT NULL,
                offset_distance DOUBLE PRECISION,
                geom GEOMETRY(POLYGON, 28992),
                PRIMARY KEY (identificatie, underpass_id)
            );
            CREATE INDEX IF NOT EXISTS idx_extended_geom_identificatie ON {t} (identificatie);
            CREATE INDEX IF NOT EXISTS idx_extended_geom_underpass_id ON {t} (underpass_id);
            CREATE INDEX IF NOT EXISTS idx_extended_geom_spatial ON {t} USING GIST (geom);
        """
    with conn.cursor() as cursor:
        cursor.execute(schema_sql.format(t=output_table))
        conn.commit()
    print("Extended geometries table ready (preserving existing data)")


def get_underpass_chunks(conn, edges_table, output_table, skipped_table, id_column) -> List[List]:
    """Get identifier chunks for chunking, excluding already processed and skipped ones."""
    with conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT DISTINCT e.{id_column}
            FROM {edges_table} e
            LEFT JOIN {output_table} eg 
                ON e.{id_column} = eg.{id_column} 
                AND e.identificatie = eg.identificatie
            LEFT JOIN {skipped_table} su
                ON e.{id_column} = su.{id_column} 
                AND e.identificatie = su.identificatie
            WHERE e.geom IS NOT NULL 
                AND eg.{id_column} IS NULL  -- Only unprocessed
                AND su.{id_column} IS NULL  -- Only non-skipped
            ORDER BY e.{id_column}
        """)
        ids = [row[0] for row in cursor.fetchall()]

    if not ids:
        print("No unprocessed features found - all work is complete!")
        return []

    print(f"Found {len(ids)} unprocessed features (excluding skipped)")

    # Create chunks with actual ID lists (not ranges)
    chunks = []
    for i in range(0, len(ids), CHUNK_SIZE):
        chunks.append(ids[i : i + CHUNK_SIZE])

    return chunks


def _build_edge_records(rows, id_column) -> dict:
    """Group raw DB rows into EdgeRecords keyed by id_column."""
    building_mode = id_column == "identificatie"
    edge_groups: dict[tuple, dict[str, list[bytes]]] = {}
    for row in rows:
        if building_mode:
            identificatie, edge_type, edge_wkb = row
            underpass_id = None
        else:
            identificatie, underpass_id, edge_type, edge_wkb = row
            underpass_id = int(underpass_id)
        key = (str(identificatie), underpass_id)
        if key not in edge_groups:
            edge_groups[key] = {"exterior": [], "shared": [], "interior": []}
        if edge_type in edge_groups[key] and edge_wkb is not None:
            edge_groups[key][edge_type].append(edge_wkb)

    records_by_id: dict = defaultdict(list)
    for (identificatie, underpass_id), edge_types in edge_groups.items():
        exterior_geoms = [
            coerce_multiline_geometry(from_wkb(bytes(w)))
            for w in edge_types["exterior"]
        ]
        movable_edges = merge_multiline_geometries(*exterior_geoms)
        shared_geoms = [
            coerce_multiline_geometry(from_wkb(bytes(w))) for w in edge_types["shared"]
        ]
        interior_geoms = [
            coerce_multiline_geometry(from_wkb(bytes(w)))
            for w in edge_types["interior"]
        ]
        fixed_edges = merge_multiline_geometries(*(shared_geoms + interior_geoms))
        record = EdgeRecord(
            identificatie=identificatie,
            underpass_id=underpass_id,
            movable_edges=movable_edges,
            fixed_edges=fixed_edges,
        )
        record_key = identificatie if building_mode else underpass_id
        records_by_id[record_key].append(record)
    return records_by_id


def process_chunk(
    chunk: List,
    chunk_num: int,
    distance: float,
    edges_table: str,
    output_table: str,
    skipped_table: str,
    id_column: str,
    db_params: dict,
) -> dict:
    """Process a chunk of features and store results in database."""
    building_mode = id_column == "identificatie"
    processed = 0
    failed = 0

    print(f"🔄 Starting chunk {chunk_num}: {len(chunk)} underpasses")

    with connect(**db_params) as conn:
        # 1. Batch-load ALL edges for this chunk in ONE query
        t0 = time.time()
        if building_mode:
            select_sql = f"""
                SELECT identificatie::text, edge_type,
                       ST_AsBinary(geom) AS edge_wkb
                FROM {edges_table}
                WHERE identificatie = ANY(%s)
                  AND geom IS NOT NULL AND NOT ST_IsEmpty(geom)
                ORDER BY identificatie, edge_type
            """
        else:
            select_sql = f"""
                SELECT identificatie::text, underpass_id, edge_type,
                       ST_AsBinary(geom) AS edge_wkb
                FROM {edges_table}
                WHERE underpass_id = ANY(%s)
                  AND geom IS NOT NULL AND NOT ST_IsEmpty(geom)
                ORDER BY identificatie, underpass_id, edge_type
            """
        with conn.cursor() as cursor:
            cursor.execute(select_sql, (chunk,))
            rows = cursor.fetchall()
        print(
            f"📥 Chunk {chunk_num}: Loaded {len(rows)} edges in {time.time() - t0:.1f}s"
        )

        # 2. Build EdgeRecords from in-memory data
        records_by_id = _build_edge_records(rows, id_column)
        del rows  # free memory

        # 3. Process each underpass purely in-memory (no more DB calls)
        batch_inserts = []
        skipped_inserts = []

        for feature_id in chunk:
            records = records_by_id.get(feature_id, [])
            if not records:
                continue

            for record in records:
                if record.movable_edges.is_empty:
                    print(
                        f"⚠️ Skipping feature {feature_id} - no movable edges found"
                    )
                    if building_mode:
                        skipped_inserts.append(
                            (record.identificatie, "no_movable_edges")
                        )
                    else:
                        skipped_inserts.append(
                            (record.identificatie, record.underpass_id, "no_movable_edges")
                        )
                    failed += 1
                    continue
                try:
                    classified = classify_polygon_from_edge_sets(
                        movable_edges=record.movable_edges,
                        fixed_edges=record.fixed_edges,
                        tolerance=1e-3,
                    )
                    polygon = offset_polygon_from_classified_polygon(
                        classified,
                        distance=distance,
                        tolerance=1e-3,
                        strategy="boolean_patch",
                    )
                    if building_mode:
                        batch_inserts.append(
                            (record.identificatie, distance, to_wkb(polygon))
                        )
                    else:
                        batch_inserts.append(
                            (
                                record.identificatie,
                                record.underpass_id,
                                distance,
                                to_wkb(polygon),
                            )
                        )
                    processed += 1
                except KeyboardInterrupt:
                    print(
                        f"🛑 Chunk {chunk_num} interrupted at feature {feature_id}"
                    )
                    break
                except InvalidInputPolygonError as e:
                    print(f"Invalid input polygon for feature {feature_id}: {e}")
                    if building_mode:
                        skipped_inserts.append(
                            (record.identificatie, "invalid_input_polygon")
                        )
                    else:
                        skipped_inserts.append(
                            (record.identificatie, record.underpass_id, "invalid_input_polygon")
                        )
                    failed += 1
                except GeometryOffsetError as e:
                    print(f"Offset failed for feature {feature_id}: {e}")
                    if building_mode:
                        skipped_inserts.append(
                            (record.identificatie, "geometry_offset_failed")
                        )
                    else:
                        skipped_inserts.append(
                            (record.identificatie, record.underpass_id, "geometry_offset_failed")
                        )
                    failed += 1
                except ValueError as e:
                    if "Polygon boundary segment was not found" in str(e):
                        print(
                            f"❌ Skipping feature {feature_id} - edge matching failed"
                        )
                        skip_reason = "edge_matching_failed"
                    else:
                        print(f"ValueError for feature {feature_id}: {e}")
                        skip_reason = "value_error"
                    if building_mode:
                        skipped_inserts.append((record.identificatie, skip_reason))
                    else:
                        skipped_inserts.append(
                            (record.identificatie, record.underpass_id, skip_reason)
                        )
                    failed += 1
                except Exception as e:
                    print(f"Error processing feature {feature_id}: {e}")
                    failed += 1
            else:
                continue
            break  # propagate KeyboardInterrupt break from inner loop

        # 4. Batch insert results in one transaction
        if batch_inserts:
            print(f"💾 Chunk {chunk_num}: Inserting {len(batch_inserts)} records...")
            if building_mode:
                insert_sql = f"""
                     INSERT INTO {output_table}
                    (identificatie, offset_distance, geom)
                    VALUES (%s, %s, ST_GeomFromWKB(%s, 28992))
                    ON CONFLICT (identificatie) DO NOTHING
                """
            else:
                insert_sql = f"""
                     INSERT INTO {output_table}
                    (identificatie, underpass_id, offset_distance, geom)
                    VALUES (%s, %s, %s, ST_GeomFromWKB(%s, 28992))
                    ON CONFLICT (identificatie, underpass_id) DO NOTHING
                """
            with conn.cursor() as cursor:
                cursor.executemany(insert_sql, batch_inserts)
            conn.commit()

        if skipped_inserts:
            print(f"💾 Chunk {chunk_num}: Recording {len(skipped_inserts)} skipped...")
            if building_mode:
                skip_sql = f"""
                     INSERT INTO {skipped_table}
                    (identificatie, skip_reason)
                    VALUES (%s, %s)
                    ON CONFLICT (identificatie) DO NOTHING
                """
            else:
                skip_sql = f"""
                     INSERT INTO {skipped_table}
                    (identificatie, underpass_id, skip_reason)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (identificatie, underpass_id) DO NOTHING
                """
            with conn.cursor() as cursor:
                cursor.executemany(skip_sql, skipped_inserts)
            conn.commit()

    print(f"🏁 Chunk {chunk_num} completed: {processed} processed, {failed} skipped")
    return {"processed": processed, "failed": failed}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        environ.setdefault(key.strip(), value.strip())


def _require_env(name: str) -> str:
    value = environ.get(name)
    if value:
        return value
    raise ValueError(f"{name} must be set.")


if __name__ == "__main__":
    sys.exit(main())
