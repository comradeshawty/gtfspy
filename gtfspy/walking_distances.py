import networkx as nx
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import sqlite3
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from ast import literal_eval


def get_nearby_nodes(walk_network, threshold_distance):
    """
    Find nearby nodes within a threshold distance.

    Parameters:
    walk_network: networkx.Graph
        Walk network graph with nodes having position (x, y) attributes.
    threshold_distance: float
        Threshold distance to consider nodes as "nearby" (in meters).

    Returns:
    dict
        A dictionary mapping each node ID to a list of nearby nodes within the threshold distance.
    """
    # Extract coordinates from the walk network
    nodes=walk_network.nodes(data=True)
    node_coords = {node: (data['lon'], data['lat']) for node, data in walk_network.nodes(data=True)}
    nodes_df=pd.DataFrame.from_dict(dict(walk_network.nodes(data=True)), orient='index')
    nodes_df['geometry'] = nodes_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    nodes_gdf = gpd.GeoDataFrame(stops, geometry='geometry', crs="EPSG:4326").to_crs(epsg=32616)
    node_coords = list(zip(nodes_gdf.geometry.x, nodes_gdf.geometry.y))
    #node_tree = cKDTree(node_coords)
    node_ids = list(node_coords.keys())
    node_positions = np.array(list(node_coords.values()))
    kd_tree = KDTree(node_positions)

    nearby_nodes = {}
    for idx, node_id in enumerate(node_ids):
      indices = kd_tree.query_ball_point(node_positions[idx], r=threshold_distance)
      nearby_nodes[node_id] = [node_ids[i] for i in indices if i != idx]
    return nearby_nodes

def precompute_walking_distances(walk_network, threshold_distance=2000):

    """
    Precompute walking distances between nearby nodes.

    Parameters:
    walk_network: networkx.Graph
        Walk network graph.
    nearby_nodes: dict
        A dictionary mapping each node to a list of nearby nodes.

    Returns:
    dict
        A dictionary mapping pairs of node IDs to the walking distance between them.
    """

    node_ids = list(walk_network.nodes)
    node_positions = np.array([(walk_network.nodes[node]['lon'], walk_network.nodes[node]['lat']) for node in node_ids])

    node_tree = cKDTree(node_positions)
    precomputed_distances = {}
    for idx, node_id in enumerate(node_ids):
        distances, indices = node_tree.query(node_positions[idx], k=len(node_ids), distance_upper_bound=threshold_distance)

        # Iterate over neighbors
        for distance, index in zip(distances, indices):
            if index != idx and distance < threshold_distance and distance > 0:
                other_node_id = node_ids[index]
                # Use a sorted tuple to prevent duplicate pairs (e.g., (1, 2) and (2, 1))
                node_pair = tuple(sorted((node_id, other_node_id)))
                precomputed_distances[node_pair] = distance

    return precomputed_distances


def save_walking_distances_to_csv(walking_distances, filename='walking_distances.csv'):
    """
    Save precomputed walking distances to a CSV file.

    Parameters:
    walking_distances: dict
        A dictionary mapping pairs of node IDs to the walking distance between them.
    filename: str
        The name of the CSV file.
    """
    data = [{'origin': origin, 'target': target, 'distance': distance} for (origin, target), distance in walking_distances.items()]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Save precomputed distances to a CSV file
#save_walking_distances_to_csv(precomputed_distances)

import sqlite3

def save_walking_distances_to_db(walking_distances, db_filename='walking_distances.db'):
    """
    Save precomputed walking distances to an SQLite database.

    Parameters:
    walking_distances: dict
        A dictionary mapping pairs of node IDs to the walking distance between them.
    db_filename: str
        The name of the SQLite database file.
    """
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Create the table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS walking_distances (
            origin INTEGER,
            target INTEGER,
            distance REAL,
            PRIMARY KEY (origin, target)
        )
    ''')
    
    # Insert the walking distances
    data = [(origin, target, distance) for (origin, target), distance in walking_distances.items()]
    cursor.executemany('INSERT OR REPLACE INTO walking_distances (origin, target, distance) VALUES (?, ?, ?)', data)
    
    conn.commit()
    conn.close()

# Save precomputed distances to SQLite database
#save_walking_distances_to_db(precomputed_distances)

def load_walking_distances_from_csv(filename='walking_distances.csv'):
    df = pd.read_csv(filename)
    return {(row['origin'], row['target']): row['distance'] for _, row in df.iterrows()}

# Load precomputed distances
#precomputed_distances = load_walking_distances_from_csv()

def load_walking_distances_from_db(db_filename='walking_distances.db'):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    cursor.execute('SELECT origin, target, distance FROM walking_distances')
    walking_distances = {(row[0], row[1]): row[2] for row in cursor.fetchall()}
    
    conn.close()
    return walking_distances

# Load precomputed distances from the database
#precomputed_distances = load_walking_distances_from_db()
