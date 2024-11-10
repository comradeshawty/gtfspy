from matplotlib import pyplot as plt
from matplotlib import rc
import sqlite3
from gtfspy.routing.helpers import get_transit_connections, get_walk_network
from gtfspy.gtfs import GTFS
from gtfspy import import_gtfs
import numpy as np
from scipy.spatial import cKDTree

from gtfspy import osm_transfers
from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler
from gtfspy.routing.node_profile_analyzer_time_and_veh_legs import NodeProfileAnalyzerTimeAndVehLegs
from collections import defaultdict
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from shapely import Point
from ast import literal_eval



def init_db(gtfs_path, db_name):
    gtfs_path='/content/drive/MyDrive/safegraph/google_transit_Birmingham_AL.zip'

    import_gtfs.import_gtfs(gtfs_path, db_name)
    gtfs_conn = sqlite3.connect(db_name)
    return GTFS(gtfs_conn)

  
def find_cbgs_to_stops(G, census_gdf_path, radius=1000):
    
    # Load Census and stops data
    census_gdf = gpd.read_file(census_gdf_path)
    stops = G.get_table("stops")
    census_gdf = census_gdf.to_crs(epsg=32616)
    
    # Convert stop coordinates to GeoDataFrame
    stops['geometry'] = stops.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    stops_gdf = gpd.GeoDataFrame(stops, geometry='geometry', crs="EPSG:4326").to_crs(epsg=32616)
    
    # Extract coordinates
    cbg_coords = np.array(list(zip(census_gdf.geometry.centroid.x, census_gdf.geometry.centroid.y)))
    stop_coords = np.array(list(zip(stops_gdf.geometry.x, stops_gdf.geometry.y)))

    # Create cKDTree for stops
    stop_tree = cKDTree(stop_coords)

    # Map each CBG to its nearby stops
    results = []
    for idx, cbg_coord in enumerate(cbg_coords):
        distance, stop_index = stop_tree.query(cbg_coord)
        stop_I = stops_gdf.iloc[stop_index]['stop_I']
        stop_name = stops_gdf.iloc[stop_index]['name']
        results.append({
            'cbg_id': census_gdf.iloc[idx]['GEOID'],
            'stop_I': [stop_I],
            'stop_name': [stop_name],
            'lat': cbg_coord[1],
            'lon': cbg_coord[0]
        })
    return pd.DataFrame(results)

def add_cbgs_as_nodes(walk_network, cbgs_to_stops, stops_gdf):
    """
    Add CBG centroids as nodes and connect them to nearby stops in the walk network.

    Parameters
    ----------
    walk_network : networkx.Graph
        The existing walk network graph.
    cbgs_to_stops : pd.DataFrame
        DataFrame mapping CBG centroids to nearby stops.
    stops_gdf : GeoDataFrame
        GeoDataFrame containing stops information.
    
    Returns
    -------
    networkx.Graph
        Updated walk network graph.
    """
    # Validate that stops_gdf is a DataFrame
    if not isinstance(stops_gdf, pd.DataFrame):
        raise TypeError("stops_gdf should be a pandas DataFrame")

    # Ensure that stop IDs are integers
    stops_gdf['stop_I'] = stops_gdf['stop_I'].astype(int)
    
    # Assign unique integer node IDs to CBGs
    max_stop_I = stops_gdf['stop_I'].max()
    
    # Extract unique CBG IDs and assign unique node IDs
    cbg_ids = cbgs_to_stops['cbg_id'].unique()
    cbg_node_ids = {cbg_id: max_stop_I + idx + 1 for idx, cbg_id in enumerate(cbg_ids)}

    # Add nodes for each CBG centroid
    for _, row in cbgs_to_stops.iterrows():
        cbg_id = row['cbg_id']
        node_id = cbg_node_ids[cbg_id]
        walk_network.add_node(node_id, lat=row['lat'], lon=row['lon'], name=str(cbg_id))

    # Add edges between CBGs and nearby stops
    for _, row in cbgs_to_stops.iterrows():
        cbg_id = row['cbg_id']
        node_id = cbg_node_ids[cbg_id]
        stop_Is = [int(sid) for sid in row['stop_I']]
        
        cbg_coord = np.array([row['lon'], row['lat']])
        for stop_I in stop_Is:
            stop_Ix = stops_gdf[stops_gdf['stop_I'] == stop_I].index[0]
            stop_coord = np.array([stops_gdf.iloc[stop_Ix].geometry.x, stops_gdf.iloc[stop_Ix].geometry.y])
            
            # Calculate Euclidean distance and add edges
            distance = np.linalg.norm(cbg_coord - stop_coord)
            walk_network.add_edge(node_id, stop_I, d_walk=distance)
            walk_network.add_edge(stop_I, node_id, d_walk=distance)
    
    return walk_network, cbg_node_ids

def compute_travel_time_matrix(G, walk_network, cbg_ids, cbg_node_ids, analysis_start_time, analysis_end_time):
    """
    Compute travel time matrix between CBG centroids using MultiObjectivePseudoCSAProfiler.

    Parameters
    ----------
    G : GTFS
        GTFS object containing transit information.
    walk_network : networkx.Graph
        Walk network graph.
    cbg_ids : list
        List of CBG IDs.
    cbg_node_ids : dict
        Mapping of CBG IDs to unique node IDs.
    analysis_start_time : int
        Start time of analysis in UNIX seconds.
    analysis_end_time : int
        End time of analysis in UNIX seconds.

    Returns
    -------
    pd.DataFrame
        Travel time matrix DataFrame.
    """
    connections = get_transit_connections(G, analysis_start_time, analysis_end_time + 2 * 3600)
    T_cbg = {}

    all_node_ids = set(walk_network.nodes())

    for idx, origin_cbg_id in enumerate(cbg_ids):
        origin_node_id = cbg_node_ids[origin_cbg_id]
        if origin_node_id not in all_node_ids:
            print(f"Origin node {origin_node_id} not found in network.")
            continue

        # Prepare targets as other CBG centroid node IDs
        target_cbg_ids = [cbg_id for cbg_id in cbg_ids if cbg_id != origin_cbg_id]
        target_node_ids = [cbg_node_ids[cbg_id] for cbg_id in target_cbg_ids]
        valid_targets = [node_id for node_id in target_node_ids if node_id in all_node_ids]

        if not valid_targets:
            print(f"No valid targets found for origin {origin_cbg_id}.")
            continue

        # Set up the profiler with the origin node
        mpCSA = MultiObjectivePseudoCSAProfiler(
            connections,
            targets=valid_targets,
            start_time_ut=analysis_start_time,
            end_time_ut=analysis_end_time,
            transfer_margin=120,  # seconds
            walk_network=walk_network,
            walk_speed=1.5,  # meters per second
            verbose=False,
            track_vehicle_legs=True,
            track_time=True
        )
        mpCSA.set_source_node(origin_node_id)  # Set the origin node

        # Run the profiler and collect results
        mpCSA.run()
        profiles = mpCSA.stop_profiles

        # Initialize the travel time dictionary for this origin
        T_cbg[origin_cbg_id] = {}

        # Iterate over target CBGs
        for target_cbg_id in cbg_ids:
            target_node_id = cbg_node_ids[target_cbg_id]
            if target_cbg_id == origin_cbg_id:
                T_cbg[origin_cbg_id][target_cbg_id] = 0
            else:
                target_profile = profiles.get(target_node_id, None)
                if target_profile:
                    labels = target_profile.get_final_optimal_labels()
                    # For each label, get the travel time
                    travel_times = [label.arrival_time_target - label.departure_time for label in labels]
                    if travel_times:
                        travel_time = min(travel_times)
                        T_cbg[origin_cbg_id][target_cbg_id] = travel_time
                    else:
                        T_cbg[origin_cbg_id][target_cbg_id] = np.nan
                else:
                    T_cbg[origin_cbg_id][target_cbg_id] = np.nan

        print(f"Processed CBG {idx+1}/{len(cbg_ids)}")

    return pd.DataFrame(T_cbg).T
