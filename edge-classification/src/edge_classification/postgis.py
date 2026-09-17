"""Database operations for edge classification."""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Tuple

from psycopg import Connection
from psycopg.sql import SQL, Identifier
from shapely import from_wkb, to_wkb
from shapely.geometry import LineString, Polygon

from edge_classification.edge_classifier import classify_edges_for_underpass


def create_geometries_cache_table(
    connection: Connection[Any],
    geometries_table: str = "underpasses.geometries",
    bag_bgt_join_table: str = "underpasses.bag_bgt_join",
    cache_table_name: str = "underpasses.geometries_cache",
    id_column: str = "underpass_id",
    geom_column: str = "geom",
    mode: str = "underpass",
) -> str:
    """
    Create an UNLOGGED table with pre-joined underpass and BGT geometry data.
    This avoids repeating the JOIN on every chunk query.

    In "building" mode the BGT geometry is not needed, so no bag_bgt_join
    happens and the bgt_geom column is left NULL.

    UNLOGGED tables are faster but not crash-safe (fine for temporary processing).

    Returns:
        Name of the cache table
    """
    print(f"Creating geometries cache table {cache_table_name}...")
    t0 = time.time()
    
    # Parse schema and table
    if '.' in cache_table_name:
        schema, table = cache_table_name.split('.', 1)
    else:
        schema = 'public'
        table = cache_table_name

    un_id = SQL("un.{col}").format(col=Identifier(id_column))
    un_geom = SQL("un.{col}").format(col=Identifier(geom_column))

    if mode == "building":
        query = SQL("""
            DROP TABLE IF EXISTS {cache_table};

            CREATE UNLOGGED TABLE {cache_table} AS
            SELECT
                {un_id}::text AS id,
                un.identificatie::text AS identificatie,
                {un_geom} AS underpass_geom,
                NULL::geometry AS bgt_geom
            FROM {geometries_table} un
            WHERE NOT ST_IsEmpty({un_geom});

            CREATE INDEX idx_geometries_cache_id ON {cache_table} (id);
            CREATE INDEX idx_geometries_cache_identificatie ON {cache_table} (identificatie);
            CREATE INDEX idx_geometries_cache_underpass_spatial ON {cache_table} USING GIST (underpass_geom);
        """).format(
            cache_table=Identifier(schema, table),
            geometries_table=Identifier(*geometries_table.split('.')),
            un_id=un_id,
            un_geom=un_geom,
        )
    else:
        query = SQL("""
            DROP TABLE IF EXISTS {cache_table};

            CREATE UNLOGGED TABLE {cache_table} AS
            SELECT
                {un_id}::text AS id,
                un.identificatie::text AS identificatie,
                {un_geom} AS underpass_geom,
                bbj.bgt_geometrie AS bgt_geom
            FROM {geometries_table} un
            JOIN {bag_bgt_join_table} bbj
                ON un.identificatie = bbj.identificatie
            WHERE NOT ST_IsEmpty({un_geom});

            CREATE INDEX idx_geometries_cache_id ON {cache_table} (id);
            CREATE INDEX idx_geometries_cache_identificatie ON {cache_table} (identificatie);
            CREATE INDEX idx_geometries_cache_underpass_spatial ON {cache_table} USING GIST (underpass_geom);
            CREATE INDEX idx_geometries_cache_bgt_spatial ON {cache_table} USING GIST (bgt_geom);
        """).format(
            cache_table=Identifier(schema, table),
            geometries_table=Identifier(*geometries_table.split('.')),
            bag_bgt_join_table=Identifier(*bag_bgt_join_table.split('.')),
            un_id=un_id,
            un_geom=un_geom,
        )
    
    with connection.cursor() as cursor:
        cursor.execute(query)
    connection.commit()
    
    # Get row count
    with connection.cursor() as cursor:
        cursor.execute(SQL("SELECT COUNT(*) FROM {cache_table}").format(
            cache_table=Identifier(schema, table)
        ))
        row_count = cursor.fetchone()[0]
    
    t1 = time.time()
    print(f"✓ Created geometries cache table with {row_count:,} records in {t1-t0:.2f}s")
    
    return cache_table_name


def create_adjacency_cache_table(
    connection: Connection[Any],
    bag_adjacency_table: str = "building_types.bag_adjacency_4",
    adjacent_geom_table: str = "underpasses.bag_bgt_join",
    cache_table_name: str = "underpasses.adjacency_cache",
    geom_column: str = "bgt_geometrie",
) -> str:
    """
    Create an UNLOGGED table with pre-joined adjacency and geometry data.
    This avoids expensive JOINs on every chunk query.

    The adjacent geometry can come from different sources:
    - underpass mode: bag_bgt_join table (bgt_geometrie column)
    - building mode:  the building table itself (geometrie column)
    
    UNLOGGED tables are faster but not crash-safe (fine for temporary processing).
    
    Returns:
        Name of the cache table
    """
    print(f"Creating adjacency cache table {cache_table_name}...")
    t0 = time.time()
    
    # Parse schema and table
    if '.' in cache_table_name:
        schema, table = cache_table_name.split('.', 1)
    else:
        schema = 'public'
        table = cache_table_name

    bag_geom = SQL("bag.{col}").format(col=Identifier(geom_column))
    
    # Drop if exists and create new
    query = SQL("""
        DROP TABLE IF EXISTS {cache_table};
        
        CREATE UNLOGGED TABLE {cache_table} AS
        SELECT 
            ba.identificatie,
            ba.adjacent_identificatie,
            {bag_geom} AS geometrie
        FROM {bag_adjacency_table} ba
        JOIN {adjacent_geom_table} bag
            ON bag.identificatie = ba.adjacent_identificatie
        WHERE NOT ST_IsEmpty({bag_geom});
                        
        CREATE INDEX idx_adjacency_cache_id ON {cache_table} (identificatie);
        CREATE INDEX idx_adjacency_cache_spatial ON {cache_table} USING GIST (geometrie);
    """).format(
        cache_table=Identifier(schema, table),
        bag_adjacency_table=Identifier(*bag_adjacency_table.split('.')),
        adjacent_geom_table=Identifier(*adjacent_geom_table.split('.')),
        bag_geom=bag_geom,
    )
    
    with connection.cursor() as cursor:
        cursor.execute(query)
    connection.commit()
    
    # Get row count
    with connection.cursor() as cursor:
        cursor.execute(SQL("SELECT COUNT(*) FROM {cache_table}").format(
            cache_table=Identifier(schema, table)
        ))
        row_count = cursor.fetchone()[0]
    
    t1 = time.time()
    print(f"✓ Created cache table with {row_count:,} adjacency records in {t1-t0:.2f}s")
    
    return cache_table_name


def drop_adjacency_cache_table(
    connection: Connection[Any],
    cache_table_name: str = "underpasses.adjacency_cache",
) -> None:
    """Drop the adjacency cache table."""
    if '.' in cache_table_name:
        schema, table = cache_table_name.split('.', 1)
    else:
        schema = 'public'
        table = cache_table_name
    
    with connection.cursor() as cursor:
        cursor.execute(SQL("DROP TABLE IF EXISTS {cache_table}").format(
            cache_table=Identifier(schema, table)
        ))
    connection.commit()
    print(f"✓ Dropped cache table {cache_table_name}")


def drop_geometries_cache_table(
    connection: Connection[Any],
    cache_table_name: str = "underpasses.geometries_cache",
) -> None:
    """Drop the geometries cache table."""
    if '.' in cache_table_name:
        schema, table = cache_table_name.split('.', 1)
    else:
        schema = 'public'
        table = cache_table_name
    
    with connection.cursor() as cursor:
        cursor.execute(SQL("DROP TABLE IF EXISTS {cache_table}").format(
            cache_table=Identifier(schema, table)
        ))
    connection.commit()
    print(f"✓ Dropped geometries cache table {cache_table_name}")

def load_all_underpass_data_for_chunk(
    connection: Connection[Any],
    underpass_ids: List[str],
    geometries_table: str,
    adjacency_table: str
) -> Dict[str, Tuple[str, Polygon, Polygon, List[Polygon]]]:
    """
    Load ALL underpass data for a chunk in TWO batch queries.
    
    Uses pre-created cache tables for fast lookups without JOINs.
    
    Args:
        connection: Database connection
        underpass_ids: List of IDs (identificatie/underpass_id) to load
        geometries_table: Name of the geometries cache table (pre-joined underpass+BGT data)
        adjacency_table: Name of the adjacency cache table (pre-joined adjacency+BAG data)
        
    Returns:
        Dict mapping id to (identificatie, underpass_geom, bgt_geom, adjacent_geoms)
    """
    # Query for underpass and BGT geometries for ALL underpasses in chunk
    t0 = time.time()
    
    query = SQL("""
        SELECT
            id,
            identificatie,
            ST_AsBinary(underpass_geom) AS underpass_wkb,
            ST_AsBinary(bgt_geom) AS bgt_wkb
        FROM {cache_table}
        WHERE id = ANY(%s)
    """).format(
        cache_table=Identifier(*geometries_table.split('.')),
    )

    
    underpass_data = {}
    with connection.cursor() as cursor:
        cursor.execute(query, (underpass_ids,))
        for id_, identificatie, underpass_wkb, bgt_wkb in cursor.fetchall():
            underpass_geom = from_wkb(underpass_wkb)
            bgt_geom = from_wkb(bgt_wkb) if bgt_wkb is not None else None
            underpass_data[str(id_)] = (identificatie, underpass_geom, bgt_geom, [])
    
    t1 = time.time()
    print(f"    ⏱️  Query 1 (underpass+BGT): {t1-t0:.2f}s for {len(underpass_data)} underpasses")
    
    # Query for ALL adjacent geometries for ALL identificaties in chunk
    identificaties = [data[0] for data in underpass_data.values()]
    
    if identificaties:
        # Use adjacency cache table (pre-joined data)
        t2 = time.time()
        adjacency_query = SQL("""
            SELECT identificatie, ST_AsBinary(geometrie) AS geom_wkb
            FROM {adjacency_table}
            WHERE identificatie = ANY(%s)
        """).format(
            adjacency_table=Identifier(*adjacency_table.split('.')),
        )
        
        adjacent_by_id = {}
        with connection.cursor() as cursor:
            cursor.execute(adjacency_query, (identificaties,))
            for identificatie, geom_wkb in cursor.fetchall():
                if identificatie not in adjacent_by_id:
                    adjacent_by_id[identificatie] = []
                adjacent_by_id[identificatie].append(from_wkb(geom_wkb))
        
        t3 = time.time()
        total_geoms = sum(len(geoms) for geoms in adjacent_by_id.values())
        print(f"    ⏱️  Query 2 (adjacency): {t3-t2:.2f}s -> {total_geoms} geometries")
        
        # Add adjacent geometries to underpass data
        for id_, (identificatie, underpass_geom, bgt_geom, _) in underpass_data.items():
            adjacent_geoms = adjacent_by_id.get(identificatie, [])
            underpass_data[id_] = (identificatie, underpass_geom, bgt_geom, adjacent_geoms)
    
    return underpass_data


@dataclass
class EdgeClassificationResult:
    """Result of edge classification with edge geometries."""
    
    underpass_id: str
    identificatie: str
    edge_type: str  # 'interior', 'exterior', or 'shared'
    geom: LineString


def load_underpass_data_from_db(
    connection: Connection[Any],
    underpass_id: int,
    geometries_table: str = "underpasses.geometries",
    bag_bgt_join_table: str = "underpasses.bag_bgt_join",
    bag_adjacency_table: str = "building_types.bag_adjacency_4",
    bag_bgt_table: str = "underpasses.bag_bgt_join",
) -> Tuple[str, Polygon, Polygon, List[Polygon]]:
    """
    Load underpass data from the database for a single underpass.
    
    Args:
        connection: Database connection
        underpass_id: ID of the underpass to process
        geometries_table: Name of the geometries table
        bag_bgt_join_table: Name of the BAG-BGT join table
        bag_adjacency_table: Name of the adjacency table
        bag_bgt_table: Table with BAG-BGT joined geometries
        
    Returns:
        Tuple of (identificatie, underpass_geom, bgt_geom, adjacent_geoms)
    """
    # Query for underpass and BGT geometries
    query = SQL("""
        SELECT
            un.identificatie::text,
            ST_AsBinary(un.geom) AS underpass_wkb,
            ST_AsBinary(bbj.bgt_geometrie) AS bgt_wkb
        FROM {geometries_table} un
        JOIN {bag_bgt_join_table} bbj
            ON un.identificatie = bbj.identificatie
        WHERE un.underpass_id = %s
            AND NOT ST_IsEmpty(un.geom)
    """).format(
        geometries_table=Identifier(*geometries_table.split('.')),
        bag_bgt_join_table=Identifier(*bag_bgt_join_table.split('.')),
    )
    
    with connection.cursor() as cursor:
        cursor.execute(query, (underpass_id,))
        row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"No data found for underpass_id {underpass_id}")
        
        identificatie, underpass_wkb, bgt_wkb = row
        underpass_geom = from_wkb(underpass_wkb)
        bgt_geom = from_wkb(bgt_wkb)
    
    # Query for adjacent geometries
    adjacency_query = SQL("""
        SELECT DISTINCT
            ST_AsBinary(bag.geometrie) AS adjacent_wkb
        FROM {bag_adjacency_table} ba
        JOIN {bag_bgt_table} bag
            ON bag.identificatie = ba.adjacent_identificatie
        WHERE ba.identificatie = %s
            AND NOT ST_IsEmpty(bag.geometrie)
    """).format(
        bag_adjacency_table=Identifier(*bag_adjacency_table.split('.')),
        bag_bgt_table=Identifier(*bag_bgt_table.split('.')),
    )
    
    adjacent_geoms = []
    with connection.cursor() as cursor:
        cursor.execute(adjacency_query, (identificatie,))
        for (adjacent_wkb,) in cursor.fetchall():
            adjacent_geoms.append(from_wkb(adjacent_wkb))
    
    return identificatie, underpass_geom, bgt_geom, adjacent_geoms


def classify_edges_from_db(
    connection: Connection[Any],
    underpass_id: int,
    grid_size: float = 0.001,
    snap_tolerance: float = 0.03,
    geometries_table: str = "underpasses.geometries",
    bag_bgt_join_table: str = "underpasses.bag_bgt_join",
    bag_adjacency_table: str = "building_types.bag_adjacency_4",
    bag_bgt_table: str = "underpasses.bag_bgt_join",
) -> List[EdgeClassificationResult]:
    """
    Classify edges for a single underpass by loading data from the database.
    
    Args:
        connection: Database connection
        underpass_id: ID of the underpass to process
        grid_size: Grid size for snapping (default: 0.001)
        snap_tolerance: Tolerance for snapping adjacent geometries (default: 0.03)
        geometries_table: Name of the geometries table
        bag_bgt_join_table: Name of the BAG-BGT join table
        bag_adjacency_table: Name of the adjacency table
        bag_bgt_table: Table with BAG-BGT joined geometries
        
    Returns:
        List of EdgeClassificationResult objects
    """
    # Load data from database
    identificatie, underpass_geom, bgt_geom, adjacent_geoms = load_underpass_data_from_db(
        connection=connection,
        underpass_id=underpass_id,
        geometries_table=geometries_table,
        bag_bgt_join_table=bag_bgt_join_table,
        bag_adjacency_table=bag_adjacency_table,
        bag_bgt_table=bag_bgt_table,
    )
    
    # Classify edges
    classified = classify_edges_for_underpass(
        underpass_id=underpass_id,
        identificatie=identificatie,
        underpass_geom=underpass_geom,
        bgt_geom=bgt_geom,
        adjacent_geoms=adjacent_geoms,
        grid_size=grid_size,
        snap_tolerance=snap_tolerance,
    )
    
    # Convert to result format
    results = []
    
    for edge in classified.interior_edges:
        results.append(EdgeClassificationResult(
            underpass_id=underpass_id,
            identificatie=identificatie,
            edge_type='interior',
            geom=edge,
        ))
    
    for edge in classified.exterior_edges:
        results.append(EdgeClassificationResult(
            underpass_id=underpass_id,
            identificatie=identificatie,
            edge_type='exterior',
            geom=edge,
        ))
    
    for edge in classified.shared_edges:
        results.append(EdgeClassificationResult(
            underpass_id=underpass_id,
            identificatie=identificatie,
            edge_type='shared',
            geom=edge,
        ))
    
    return results


def write_edges_to_db(
    connection: Connection[Any],
    edges: List[EdgeClassificationResult],
    edges_table: str,
    create_table: bool = False,
) -> None:
    """
    Write classified edges to the database.
    
    Args:
        connection: Database connection
        edges: List of EdgeClassificationResult objects to write
        edges_table: Name of the output table
        create_table: Whether to create/recreate the table
    """
    if create_table:
        # Create table if requested
        create_query = SQL("""
            DROP TABLE IF EXISTS {edges_table} CASCADE;
            
            CREATE TABLE {edges_table} (
                edge_id SERIAL PRIMARY KEY,
                identificatie TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                geom GEOMETRY(LineString, 28992)
            );
            
            CREATE INDEX IF NOT EXISTS idx_edges_identificatie
                ON {edges_table} (identificatie);
            
            CREATE INDEX IF NOT EXISTS idx_edges_edge_type
                ON {edges_table} (edge_type);
            
            CREATE INDEX IF NOT EXISTS idx_edges_geom
                ON {edges_table} USING GIST (geom);
        """).format(edges_table=Identifier(*edges_table.split('.')))
        
        with connection.cursor() as cursor:
            cursor.execute(create_query)
        connection.commit()
    
    # Insert edges using batch insert for performance
    if edges:
        insert_query = SQL("""
            INSERT INTO {edges_table} (identificatie, edge_type, geom)
            VALUES (%s, %s, ST_GeomFromWKB(%s, 28992))
        """).format(edges_table=Identifier(*edges_table.split('.')))
        
        # Prepare all data for batch insert
        edge_data = [
            (edge.identificatie, edge.edge_type, to_wkb(edge.geom))
            for edge in edges
        ]
        
        with connection.cursor() as cursor:
            cursor.executemany(insert_query, edge_data)
        
        # Note: Caller is responsible for commit to allow batching multiple writes


def get_all_underpass_ids(
    connection: Connection[Any],
    geometries_table: str = "underpasses.geometries",
) -> List[int]:
    """
    Get all underpass IDs from the database.
    
    Args:
        connection: Database connection
        geometries_table: Name of the geometries table
        
    Returns:
        List of underpass IDs
    """
    query = SQL("""
        SELECT DISTINCT identificatie
        FROM {geometries_table}
        WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom)
        ORDER BY identificatie
    """).format(geometries_table=Identifier(*geometries_table.split('.')))
    
    with connection.cursor() as cursor:
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]


def get_unprocessed_underpass_ids(
    connection: Connection[Any],
    geometries_table: str = "underpasses.geometries",
    edges_table: str = "underpasses.edges",
    id_column: str = "underpass_id",
    geom_column: str = "geom",
) -> List[str]:
    """
    Get IDs that haven't been processed yet.
    
    Args:
        connection: Database connection
        geometries_table: Name of the geometries table
        edges_table: Name of the edges table
        id_column: Name of the identifier column in both tables
        geom_column: Name of the geometry column in the geometries table
        
    Returns:
        List of unprocessed IDs
    """
    un_id = SQL("un.{col}").format(col=Identifier(id_column))
    e_id = SQL("e.{col}").format(col=Identifier(id_column))
    un_geom = SQL("un.{col}").format(col=Identifier(geom_column))

    query = SQL("""
        SELECT DISTINCT {un_id}::text
        FROM {geometries_table} un
        LEFT JOIN {edges_table} e
            ON {un_id}::text = {e_id}::text
        WHERE {un_geom} IS NOT NULL 
            AND NOT ST_IsEmpty({un_geom})
            AND {e_id} IS NULL
        ORDER BY 1
    """).format(
        geometries_table=Identifier(*geometries_table.split('.')),
        edges_table=Identifier(*edges_table.split('.')),
        un_id=un_id,
        e_id=e_id,
        un_geom=un_geom,
    )
    
    with connection.cursor() as cursor:
        cursor.execute(query)
        return [str(row[0]) for row in cursor.fetchall()]
