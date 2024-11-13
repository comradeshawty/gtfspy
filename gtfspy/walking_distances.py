import networkx as nx
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import sqlite3

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
    node_coords = {node: (data['lon'], data['lon']) for node, data in walk_network.nodes(data=True)}
    node_ids = list(node_coords.keys())
    node_positions = np.array(list(node_coords.values()))

    # Use KDTree for efficient distance calculation
    node_tree = KDTree(node_positions)

    # Find all pairs of nodes within the threshold distance
    nearby_nodes = {}
    for idx, node_id in enumerate(node_ids):
        distances, indices = node_tree.query(node_positions[idx], k=len(node_ids), distance_upper_bound=threshold_distance)
        nearby_nodes[node_id] = [node_ids[i] for i in indices if distances[i] < threshold_distance and distances[i] > 0]


    return nearby_nodes
def precompute_walking_distances(walk_network, nearby_nodes):
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
    walking_distances = {}

    # Iterate over each node and its nearby nodes
    for origin_node, targets in nearby_nodes.items():
        for target_node in targets:
            if (origin_node, target_node) not in walking_distances:
                try:
                    # Compute the shortest path distance using Dijkstra's algorithm
                    distance = nx.shortest_path_length(walk_network, source=origin_node, target=target_node, weight='distance')
                    walking_distances[(origin_node, target_node)] = distance
                    walking_distances[(target_node, origin_node)] = distance  # Assume distance is symmetric
                except nx.NetworkXNoPath:
                    # In case there is no path between nodes
                    walking_distances[(origin_node, target_node)] = float('inf')
                    walking_distances[(target_node, origin_node)] = float('inf')

    return walking_distances

import pandas as pd

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


