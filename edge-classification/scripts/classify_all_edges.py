#!/usr/bin/env python3
"""
Classify edges for all underpasses in parallel.

This script processes all underpasses from the database, classifying their edges
into interior, exterior, and shared types using parallel processing.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from os import environ
from pathlib import Path
import sys
import time
from typing import Dict, List

from psycopg import connect

from edge_classification.postgis import (
    classify_edges_for_underpass,
    create_adjacency_cache_table,
    create_geometries_cache_table,
    drop_adjacency_cache_table,
    drop_geometries_cache_table,
    EdgeClassificationResult,
    get_unprocessed_underpass_ids,
    load_all_underpass_data_for_chunk,
    write_edges_to_db,
)


ENV_PATH = Path(".env")
CHUNK_SIZE = 1000


def _load_dotenv(path: Path) -> None:
    """Load environment variables from .env file if it exists."""
    if not path.exists():
        return
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                environ.setdefault(key.strip(), value.strip())


def _require_env(key: str) -> str:
    """Get required environment variable or raise error."""
    value = environ.get(key)
    if not value:
        raise ValueError(f"{key} environment variable must be set")
    return value


def setup_edges_table(db_params: Dict[str, str], edges_table: str) -> None:
    """Create the edges table if it doesn't exist."""
    # Parse schema and table name from edges_table
    if '.' in edges_table:
        schema, table = edges_table.split('.', 1)
    else:
        schema = 'public'
        table = edges_table
    
    with connect(**db_params) as conn:
        # Check if table exists
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s
                    AND table_name = %s
                )
            """, (schema, table))
            table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print(f"Creating edges table {edges_table}...")
            write_edges_to_db(conn, [], edges_table=edges_table, create_table=True)
            print("✓ Table created")


def get_underpass_chunks(db_params: Dict[str, str], geometries_table: str, edges_table: str, id_column: str, geom_column: str) -> List[List[str]]:
    """Get chunks of unprocessed underpass IDs."""
    with connect(**db_params) as conn:
        underpass_ids = get_unprocessed_underpass_ids(
            conn,
            geometries_table=geometries_table,
            edges_table=edges_table,
            id_column=id_column,
            geom_column=geom_column,
        )
    
    if not underpass_ids:
        return []
    
    # Split into chunks
    chunks = []
    for i in range(0, len(underpass_ids), CHUNK_SIZE):
        chunks.append(underpass_ids[i:i + CHUNK_SIZE])
    
    return chunks


def process_chunk(
    chunk: List[str],
    chunk_num: int,
    grid_size: float,
    snap_tolerance: float,
    edges_table: str,
    mode: str,
    db_params: Dict[str, str],
    adjacency_cache_table: str | None = None,
    geometries_cache_table: str | None = None,
) -> Dict[str, int]:
    """
    Process a chunk of underpasses.
    
    Args:
        chunk: List of underpass IDs to process
        chunk_num: Chunk number for logging
        grid_size: Grid size for snapping
        snap_tolerance: Tolerance for snapping adjacent geometries
        edges_table: Name of the edges output table
        db_params: Database connection parameters
        adjacency_cache_table: Name of pre-created adjacency cache table (optional)
        geometries_cache_table: Name of pre-created geometries cache table (optional)
        
    Returns:
        Dictionary with processing statistics
    """
    successful = 0
    failed = 0
    failed_ids = []
    
    print(f"🔄 Chunk {chunk_num}: Starting {len(chunk)} underpasses")
    
    with connect(**db_params) as conn:
        # 1. Batch-load ALL data for this chunk
        t_load_start = time.time()
        underpass_data = load_all_underpass_data_for_chunk(
            connection=conn,
            underpass_ids=chunk,  
            geometries_table=geometries_cache_table,  
            adjacency_table=adjacency_cache_table,
        )
        t_load = time.time() - t_load_start
        print(f"📥 Chunk {chunk_num}: Loaded data for {len(underpass_data)} underpasses in {t_load:.2f}s")
        
        # 2. Process each underpass in-memory (no more DB calls)
        t_process_start = time.time()
        all_edges = []
        
        for underpass_id in chunk:
            if underpass_id not in underpass_data:
                failed += 1
                failed_ids.append(underpass_id)
                print(f"  ✗ Chunk {chunk_num}: No data for underpass {underpass_id}")
                continue
            
            try:
                identificatie, underpass_geom, bgt_geom, adjacent_geoms = underpass_data[underpass_id]
                
                # Classify edges for this underpass
                classified = classify_edges_for_underpass(
                    underpass_id=underpass_id,
                    identificatie=identificatie,
                    underpass_geom=underpass_geom,
                    bgt_geom=bgt_geom,
                    adjacent_geoms=adjacent_geoms,
                    grid_size=grid_size,
                    snap_tolerance=snap_tolerance,
                    mode=mode,
                )
                
                # Convert to result format
                for edge in classified.interior_edges:
                    all_edges.append(EdgeClassificationResult(
                        underpass_id=underpass_id,
                        identificatie=identificatie,
                        edge_type='interior',
                        geom=edge,
                    ))
                
                for edge in classified.exterior_edges:
                    all_edges.append(EdgeClassificationResult(
                        underpass_id=underpass_id,
                        identificatie=identificatie,
                        edge_type='exterior',
                        geom=edge,
                    ))
                
                for edge in classified.shared_edges:
                    all_edges.append(EdgeClassificationResult(
                        underpass_id=underpass_id,
                        identificatie=identificatie,
                        edge_type='shared',
                        geom=edge,
                    ))
                
                successful += 1
                if successful % 10 == 0:  # Log every 10 underpasses
                    print(f"  ✓ Chunk {chunk_num}: Processed {successful}/{len(chunk)} underpasses...")
                
            except Exception as e:
                failed += 1
                failed_ids.append(underpass_id)
                print(f"  ✗ Chunk {chunk_num}: Failed underpass {underpass_id}: {e}")
        
        t_process = time.time() - t_process_start
        print(f"⚙️  Chunk {chunk_num}: Processed {successful} underpasses in {t_process:.2f}s ({t_process/len(chunk):.3f}s per underpass)")
        
        # 3. Batch write all edges at once
        t_write = 0
        if all_edges:
            t_write_start = time.time()
            print(f"💾 Chunk {chunk_num}: Writing {len(all_edges)} edges to database...")
            write_edges_to_db(conn, all_edges, edges_table=edges_table)
            conn.commit()
            t_write = time.time() - t_write_start
        
        print(f"✓ Chunk {chunk_num}: Complete - {successful} successful, {failed} failed")
        print(f"⏱️  Chunk {chunk_num} timing: Load {t_load:.2f}s | Process {t_process:.2f}s | Write {t_write:.2f}s | Total {t_load+t_process+t_write:.2f}s")
    
    return {
        'successful': successful,
        'failed': failed,
        'failed_ids': failed_ids,
    }


def main() -> int:
    """Main entry point."""
    _load_dotenv(ENV_PATH)
    
    # Get configuration
    grid_size = float(environ.get("EDGE_CLASSIFICATION_GRID_PRECISION", "0.001"))
    snap_tolerance = float(environ.get("EDGE_CLASSIFICATION_SNAP_TOLERANCE", "0.03"))
    max_workers = int(environ.get("EDGE_CLASSIFICATION_MAX_WORKERS", "4"))
    edges_table = environ.get("EDGE_CLASSIFICATION_EDGES_TABLE", "underpasses.edges")
    geometries_table = environ.get(
        "EDGE_CLASSIFICATION_GEOMETRIES_TABLE", "underpasses.geometries"
    )
    bag_bgt_join_table = environ.get(
        "EDGE_CLASSIFICATION_BAG_BGT_JOIN_TABLE", "underpasses.bag_bgt_join"
    )
    bag_adjacency_table = environ.get(
        "EDGE_CLASSIFICATION_BAG_ADJACENCY_TABLE", "building_types.bag_adjacency_4"
    )
    geometries_cache_table = environ.get(
        "EDGE_CLASSIFICATION_GEOMETRIES_CACHE_TABLE", "underpasses.geometries_cache"
    )
    adjacency_cache_table = environ.get(
        "EDGE_CLASSIFICATION_ADJACENCY_CACHE_TABLE", "underpasses.adjacency_cache"
    )
    id_column = environ.get("EDGE_CLASSIFICATION_ID_COLUMN", "underpass_id")
    geom_column = environ.get("EDGE_CLASSIFICATION_GEOM_COLUMN", "geom")
    mode = environ.get("EDGE_CLASSIFICATION_MODE", "underpass")
    
    # Database connection parameters
    db_params = {
        "host": _require_env("EDGE_CLASSIFICATION_DB_HOST"),
        "port": int(_require_env("EDGE_CLASSIFICATION_DB_PORT")),
        "dbname": _require_env("EDGE_CLASSIFICATION_DB_NAME"),
        "user": _require_env("EDGE_CLASSIFICATION_DB_USER"),
        "password": environ.get("EDGE_CLASSIFICATION_DB_PASSWORD", ""),
    }
    
    print("=" * 80)
    print("Edge Classification for Underpass Detection")
    print("=" * 80)
    print(f"Grid size (precision): {grid_size}")
    print(f"Snap tolerance: {snap_tolerance}")
    print(f"Max workers: {max_workers}")
    print()
    print("Setting up database...")
    print()
    
    # Setup database
    setup_edges_table(db_params, edges_table)
    print(f"Output table: {edges_table}")
    
    # Create cache tables (one-time expensive JOINs)
    adjacency_cache_table_name = None
    geometries_cache_table_name = None

    with connect(**db_params) as conn:
        # Create geometries cache (underpass + BGT JOIN)
        geometries_cache_table_name = create_geometries_cache_table(
            conn,
            geometries_table=geometries_table,
            bag_bgt_join_table=bag_bgt_join_table,
            cache_table_name=geometries_cache_table,
            id_column=id_column,
            geom_column=geom_column,
            mode=mode,
        )
        print()
        
        # Create adjacency cache (adjacency + geometry JOIN)
        if mode == "building":
            adjacency_cache_table_name = create_adjacency_cache_table(
                conn,
                bag_adjacency_table=bag_adjacency_table,
                adjacent_geom_table=geometries_table,
                cache_table_name=adjacency_cache_table,
                geom_column=geom_column,
            )
        else:
            adjacency_cache_table_name = create_adjacency_cache_table(
                conn,
                bag_adjacency_table=bag_adjacency_table,
                adjacent_geom_table=bag_bgt_join_table,
                cache_table_name=adjacency_cache_table,
                geom_column="bgt_geometrie",
            )
    
    print("✓ Cache tables created")
    print("✓ Database setup complete")
    print()
    
    # Get chunks to process
    underpass_chunks = get_underpass_chunks(db_params, geometries_table, edges_table, id_column, geom_column)
    
    if not underpass_chunks:
        print("✓ All underpasses have been processed!")
        return 0
    print(f"✓ Retrieved {len(underpass_chunks)} chunks of underpasses to process")

    total_underpasses = sum(len(chunk) for chunk in underpass_chunks)
    print(f"Found {total_underpasses} unprocessed underpasses in {len(underpass_chunks)} chunks")
    print(f"Using {max_workers} parallel workers")
    print()
    
    # Process chunks in parallel
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    all_failed_ids = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_chunk = {
            executor.submit(
                process_chunk,
                chunk,
                chunk_num + 1,
                grid_size,
                snap_tolerance,
                edges_table,
                mode,
                db_params,
                adjacency_cache_table_name,
                geometries_cache_table_name,
            ): (chunk, chunk_num + 1)
            for chunk_num, chunk in enumerate(underpass_chunks)
        }
        
        # Process completed chunks
        completed_chunks = 0
        for future in as_completed(future_to_chunk):
            chunk, chunk_num = future_to_chunk[future]
            
            try:
                result = future.result()
                total_successful += result['successful']
                total_failed += result['failed']
                all_failed_ids.extend(result['failed_ids'])
                
                completed_chunks += 1
                percent = (completed_chunks / len(underpass_chunks)) * 100
                
                print(
                    f"✓ Chunk {chunk_num}/{len(underpass_chunks)} ({percent:.1f}%): "
                    f"{result['successful']} successful, {result['failed']} failed"
                )
                
            except Exception as e:
                print(f"✗ Chunk {chunk_num} failed completely: {e}")
                total_failed += len(chunk)
    
    # Print summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("Processing Complete")
    print("=" * 80)
    print(f"Total underpasses processed: {total_successful + total_failed}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Average time per underpass: {elapsed / (total_successful + total_failed):.3f} seconds")
    
    if all_failed_ids:
        print()
        print(f"Failed underpass IDs: {all_failed_ids[:20]}")
        if len(all_failed_ids) > 20:
            print(f"... and {len(all_failed_ids) - 20} more")
    
    # Cleanup: drop the cache tables
    print()
    with connect(**db_params) as conn:
        if geometries_cache_table_name:
            drop_geometries_cache_table(conn, geometries_cache_table_name)
        if adjacency_cache_table_name:
            drop_adjacency_cache_table(conn, adjacency_cache_table_name)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
