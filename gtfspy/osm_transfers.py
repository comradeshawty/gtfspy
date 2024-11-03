import os

import networkx as nx
import pandas
import osmnx as ox
from gtfspy.gtfs import GTFS
from gtfspy.util import wgs84_distance

from warnings import warn

from geoindex import GeoGridIndex, GeoPoint


import osmnx as ox
import networkx as nx

def add_walk_distances_to_db_python(gtfs, place, network_type,cutoff_distance_m=1000):
    """
    Computes the walk paths between stops and updates these to the gtfs database.

    Parameters
    ----------
    gtfs: gtfspy.GTFS or str
        A GTFS object or a string representation.
    osm_path: str
        path to the OpenStreetMap file
    cutoff_distance_m: number
        maximum allowed distance in meters

    Returns
    -------
    None
    """
    if isinstance(gtfs, str):
        gtfs = GTFS(gtfs)
    assert isinstance(gtfs, GTFS)

    print("Reading in walk network")
    # Use OSMnx to load the walkable street network from the OSM file
    walk_network = ox.graph_from_place(place, network_type)
    print("Matching stops to the OSM network")
    # Use the previously modified function that matches GTFS stops to OSM nodes
    stop_I_to_nearest_osm_node, stop_I_to_nearest_osm_node_distance = match_stops_to_nodes(gtfs, walk_network)

    transfers = gtfs.get_straight_line_transfer_distances()

    from_I_to_to_stop_Is = {stop_I: set() for stop_I in stop_I_to_nearest_osm_node}
    for transfer_tuple in transfers.itertuples():
        from_I = transfer_tuple.from_stop_I
        to_I = transfer_tuple.to_stop_I
        from_I_to_to_stop_Is[from_I].add(to_I)

    print("Computing walking distances")
    # Iterate over each stop pair
    for from_I, to_stop_Is in from_I_to_to_stop_Is.items():
        from_node = stop_I_to_nearest_osm_node[from_I]
        from_dist = stop_I_to_nearest_osm_node_distance[from_I]

        # Calculate shortest paths from the 'from_node' using Dijkstra's algorithm with OSMnx
        shortest_paths = nx.single_source_dijkstra_path_length(
            walk_network, 
            from_node, 
            cutoff=cutoff_distance_m - from_dist, 
            weight="length"
        )

        # Loop through each destination stop to calculate total walking distances
        for to_I in to_stop_Is:
            to_distance = stop_I_to_nearest_osm_node_distance[to_I]
            to_node = stop_I_to_nearest_osm_node[to_I]
            osm_distance = shortest_paths.get(to_node, float('inf'))
            total_distance = from_dist + osm_distance + to_distance

            # Get the straight-line distance from GTFS
            from_stop_I_transfers = transfers[transfers['from_stop_I'] == from_I]
            straight_distance = from_stop_I_transfers[from_stop_I_transfers["to_stop_I"] == to_I]["d"].values[0]

            # Ensure calculated distance is accurate (allow a small margin)
            assert (straight_distance < total_distance + 2)  # Allow a max of 2 meters in the calculations

            # If the total walking distance is within the cutoff, update the GTFS database
            if total_distance <= cutoff_distance_m:
                gtfs.conn.execute(f"UPDATE stop_distances SET d_walk = {int(total_distance)} "
                                  f"WHERE from_stop_I={from_I} AND to_stop_I={to_I}")

    # Commit the changes to the database
    gtfs.conn.commit()

def match_stops_to_nodes(gtfs, walk_network):
    """
    Parameters
    ----------
    gtfs : a GTFS object
    walk_network : networkx.Graph

    Returns
    -------
    stop_I_to_node: dict
        maps stop_I to closest walk_network node
    stop_I_to_dist: dict
        maps stop_I to the distance to the closest walk_network node
    """
    # Extract node coordinates from the OSMnx walkable network
    network_nodes = {node: (data['y'], data['x']) for node, data in walk_network.nodes(data=True)}
    
    # Get the list of stop_I and their coordinates from the GTFS data
    stop_Is = set(gtfs.get_straight_line_transfer_distances()['from_stop_I'])
    stops_df = gtfs.stops()

    # Initialize dictionaries to store results
    stop_I_to_node = {}
    stop_I_to_dist = {}

    # Iterate through all stops in the GTFS data
    for stop_I in stop_Is:
        # Get stop latitude and longitude
        stop_lat = float(stops_df[stops_df.stop_I == stop_I].lat)
        stop_lon = float(stops_df[stops_df.stop_I == stop_I].lon)

        # Find the nearest network node to the stop using OSMnx
        nearest_node = ox.distance.nearest_nodes(walk_network, stop_lon, stop_lat)
        nearest_node_lat = network_nodes[nearest_node][0]
        nearest_node_lon = network_nodes[nearest_node][1]

        # Calculate the Euclidean distance between the stop and the nearest network node
        dist = ox.distance.great_circle_vec(stop_lat, stop_lon, nearest_node_lat, nearest_node_lon)  # in meters

        # Store the results
        stop_I_to_node[stop_I] = nearest_node
        stop_I_to_dist[stop_I] = dist

    return stop_I_to_node, stop_I_to_dist


OSM_HIGHWAY_WALK_TAGS = {"trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary",
                         "tertiary_link", "unclassified", "residential", "living_street", "road", "pedestrian", "path",
                         "cycleway", "footway"}


#def create_walk_network_from_osm(place,network_type):
  #walk_network = ox.graph_from_place(place, network_type=network_type)
  #walk_network = ox.project_graph(walk_network)
  #meters_per_minute = 4.5 * 1000 / 60
  
  #for u, v, k, data in walk_network.edges(data=True, keys=True):
    #data['time'] = data['length'] / meters_per_minute
    #return walk_network
