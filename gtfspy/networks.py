import networkx
import pandas as pd
from math import isnan
from gtfspy import route_types
from gtfspy.util import wgs84_distance, graph_node_attrs
from warnings import warn
from ast import literal_eval
import geopandas as gpd
from shapely import Point

ALL_STOP_TO_STOP_LINK_ATTRIBUTES = [
    "capacity_estimate", "duration_min", "duration_max",
    "duration_median", "duration_avg", "n_vehicles", "route_types",
    "d", "distance_shape",
    "route_I_counts"
]

DEFAULT_STOP_TO_STOP_LINK_ATTRIBUTES = [
    "n_vehicles", "duration_avg",
    "d", "route_I_counts"
]

def get_walk_network(gtfs, pois=False, max_link_distance=1000):
    """
    Parameters
    ----------
    gtfs: gtfspy.GTFS
        GTFS object containing transit information.
    pois: bool, optional
        Whether to include Points of Interest (POIs) in the walk network.
    max_link_distance: int, optional
        Maximum distance for walking links (in meters).

    Returns
    -------
    walk_network: networkx.Graph
        The generated walking network graph.
    """
    assert isinstance(gtfs, GTFS), "Input must be a GTFS object"
    return walk_transfer_stop_to_stop_network(gtfs, max_link_distance=max_link_distance)

def walk_transfer_stop_to_stop_network(gtfs, max_link_distance=1000):
    """
    Construct the walk network.
    If OpenStreetMap-based walking distances have been computed, then those are used as the distance.
    Otherwise, the great circle distances ("d") is used.

    Parameters
    ----------
    gtfs: gtfspy.GTFS
        The GTFS object with stop and distance data.
    max_link_distance: int, optional
        All walking transfers with a distance longer than this limit
        (expressed in meters) will be omitted.

    Returns
    -------
    net: networkx.DiGraph
        Edges have attributes:
            d: straight-line distance between stops
            d_walk: distance along the road/tracks if available
    """
    net = nx.Graph()
    stops = gtfs.get_table("stops")
    
    # Convert stops to GeoDataFrame and change CRS to EPSG:32616
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops['lon'], stops['lat']),
        crs="EPSG:4326"
    ).to_crs(epsg=32616)

    # Add stops as nodes to the network
    for _, row in stops_gdf.iterrows():
        net.add_node(row['stop_I'], x=row.geometry.x, y=row.geometry.y)

    # Retrieve pre-computed distances (e.g., OSM-based or fallback)
    stop_distances = gtfs.get_table("stop_distances")
    osm_distances_available = stop_distances['d_walk'].notna().all()

    if not osm_distances_available:
        warnings.warn(
            "Warning: OpenStreetMap-based walking distances have not been computed, using Euclidean distances instead. "
            "Ignore this warning if running unit tests."
        )

    # Iterate over each stop pair and add edges based on distance
    for stop_distance_tuple in stop_distances.itertuples():
        from_node = stop_distance_tuple.from_stop_I
        to_node = stop_distance_tuple.to_stop_I

        if osm_distances_available:
            d_walk = stop_distance_tuple.d_walk
            if d_walk > max_link_distance or np.isnan(d_walk):
                continue
            data = {'d': stop_distance_tuple.d, 'd_walk': d_walk}
        else:
            d = stop_distance_tuple.d
            if d > max_link_distance:
                continue
            data = {'d_walk': d}
        
        net.add_edge(from_node, to_node, **data)

    return net

def stop_to_stop_network_for_route_type(gtfs,
                                        route_type,
                                        link_attributes=None,
                                        start_time_ut=None,
                                        end_time_ut=None,pois=False):
    """
    Get a stop-to-stop network describing a single mode of travel.

    Parameters
    ----------
    gtfs : gtfspy.GTFS
    route_type : int
        See gtfspy.route_types.TRANSIT_ROUTE_TYPES for the list of possible types.
    link_attributes: list[str], optional
        defaulting to use the following link attributes:
            "n_vehicles" : Number of vehicles passed
            "duration_min" : minimum travel time between stops
            "duration_max" : maximum travel time between stops
            "duration_median" : median travel time between stops
            "duration_avg" : average travel time between stops
            "d" : distance along straight line (wgs84_distance)
            "distance_shape" : minimum distance along shape
            "capacity_estimate" : approximate capacity passed through the stop
            "route_I_counts" : dict from route_I to counts
    start_time_ut: int
        start time of the time span (in unix time)
    end_time_ut: int
        end time of the time span (in unix time)

    Returns
    -------
    net: networkx.DiGraph
        A directed graph Directed graph
    """
    if link_attributes is None:
        link_attributes = DEFAULT_STOP_TO_STOP_LINK_ATTRIBUTES
    assert(route_type in route_types.TRANSIT_ROUTE_TYPES)
    stops_dataframe = gtfs.get_stops_for_route_type(route_type)
    net = networkx.DiGraph()
    if pois==False:
        _add_stops_to_net(net, stops_dataframe)
    else:                                    
        stops_gdf=pd.read_csv('/content/drive/MyDrive/safegraph/stops_gdf')                                    
        _add_stops_and_pois_to_net(net, stops_dataframe, stops_gdf)

    events_df = gtfs.get_transit_events(start_time_ut=start_time_ut,
                                        end_time_ut=end_time_ut,
                                        route_type=route_type)
    if len(net.nodes()) < 2:
        assert events_df.shape[0] == 0

    # group events by links, and loop over them (i.e. each link):
    link_event_groups = events_df.groupby(['from_stop_I', 'to_stop_I'], sort=False)
    for key, link_events in link_event_groups:
        from_stop_I, to_stop_I = key
        assert isinstance(link_events, pd.DataFrame)
        # 'dep_time_ut' 'arr_time_ut' 'shape_id' 'route_type' 'trip_I' 'duration' 'from_seq' 'to_seq'
        if link_attributes is None:
            net.add_edge(from_stop_I, to_stop_I)
        else:
            link_data = {}
            if "duration_min" in link_attributes:
                link_data['duration_min'] = float(link_events['duration'].min())
            if "duration_max" in link_attributes:
                link_data['duration_max'] = float(link_events['duration'].max())
            if "duration_median" in link_attributes:
                link_data['duration_median'] = float(link_events['duration'].median())
            if "duration_avg" in link_attributes:
                link_data['duration_avg'] = float(link_events['duration'].mean())
            # statistics on numbers of vehicles:
            if "n_vehicles" in link_attributes:
                link_data['n_vehicles'] = int(link_events.shape[0])
            if "capacity_estimate" in link_attributes:
                link_data['capacity_estimate'] = route_types.ROUTE_TYPE_TO_APPROXIMATE_CAPACITY[route_type] \
                                                 * int(link_events.shape[0])
            if "d" in link_attributes:
                from_lat = graph_node_attrs(net, from_stop_I)['lat']
                from_lon = graph_node_attrs(net, from_stop_I)['lon']
                to_lat = graph_node_attrs(net, to_stop_I)['lat']
                to_lon = graph_node_attrs(net, to_stop_I)['lon']
                distance = wgs84_distance(from_lat, from_lon, to_lat, to_lon)
                link_data['d'] = int(distance)
            if "distance_shape" in link_attributes:
                assert "shape_id" in link_events.columns.values
                found = None
                for i, shape_id in enumerate(link_events["shape_id"].values):
                    if shape_id is not None:
                        found = i
                        break
                if found is None:
                    link_data["distance_shape"] = None
                else:
                    link_event = link_events.iloc[found]
                    distance = gtfs.get_shape_distance_between_stops(
                        link_event["trip_I"],
                        int(link_event["from_seq"]),
                        int(link_event["to_seq"])
                    )
                    link_data['distance_shape'] = distance
            if "route_I_counts" in link_attributes:
                link_data["route_I_counts"] = link_events.groupby("route_I").size().to_dict()
            net.add_edge(from_stop_I, to_stop_I, **link_data)
    return net


def stop_to_stop_networks_by_type(gtfs):
    """
    Compute stop-to-stop networks for all travel modes (route_types).

    Parameters
    ----------
    gtfs: gtfspy.GTFS

    Returns
    -------
    dict: dict[int, networkx.DiGraph]
        keys should be one of route_types.ALL_ROUTE_TYPES (i.e. GTFS route_types)
    """
    route_type_to_network = dict()
    for route_type in route_types.ALL_ROUTE_TYPES:
        if route_type == route_types.WALK:
            net = walk_transfer_stop_to_stop_network(gtfs)
        else:
            net = stop_to_stop_network_for_route_type(gtfs, route_type)
        route_type_to_network[route_type] = net
    assert len(route_type_to_network) == len(route_types.ALL_ROUTE_TYPES)
    return route_type_to_network

def combined_stop_to_stop_transit_network(gtfs, start_time_ut=None, end_time_ut=None):
    """
    Compute stop-to-stop networks for all travel modes and combine them into a single network.
    The modes of transport are encoded to a single network.
    The network consists of multiple links corresponding to each travel mode.
    Walk mode is not included.

    Parameters
    ----------
    gtfs: gtfspy.GTFS

    Returns
    -------
    net: networkx.MultiDiGraph
        keys should be one of route_types.TRANSIT_ROUTE_TYPES (i.e. GTFS route_types)
    """
    multi_di_graph = networkx.MultiDiGraph()
    for route_type in route_types.TRANSIT_ROUTE_TYPES:
        graph = stop_to_stop_network_for_route_type(gtfs, route_type,
                                                    start_time_ut=start_time_ut, end_time_ut=end_time_ut)
        for from_node, to_node, data in graph.edges(data=True):
            data['route_type'] = route_type
        multi_di_graph.add_edges_from(graph.edges(data=True))
        multi_di_graph.add_nodes_from(graph.nodes(data=True))
    return multi_di_graph

def _add_stops_and_pois_to_net(net, stops, stops_gdf):
    """
    Add stop nodes and their nearby POIs as nodes in the network, connecting stops to POIs with weighted edges.

    Parameters
    ----------
    net: networkx.Graph
    stops: pandas.DataFrame
        DataFrame containing stop details.
    stops_gdf: pandas.DataFrame
        DataFrame containing stop details, nearby POIs, and distances.
    """
    for stop in stops.itertuples():
        # Add stop node
        stop_data = {
            "lat": stop.lat,
            "lon": stop.lon,
            "name": stop.name
        }
        net.add_node(stop.stop_I, **stop_data)
    for stop in stops_gdf.itertuples():
      stop_row = stops_gdf.loc[stops_gdf['stop_id'] == stop.stop_I]
      if not stop_row.empty:
          # Use literal_eval to convert strings back to lists if needed
          nearby_pois = stop_row['nearby_pois'].values[0]
          if isinstance(nearby_pois, str):
              nearby_pois = literal_eval(nearby_pois)

          nearby_distances = stop_row['nearby_distances'].values[0]
          if isinstance(nearby_distances, str):
              nearby_distances = literal_eval(nearby_distances)

          # Now iterate through nearby_pois and add them as nodes and edges
          for poi, distance in zip(nearby_pois, nearby_distances):
              # Add each POI as a node if not already present
              if poi not in net:
                  net.add_node(poi, type="POI")  # Customize attributes for POIs as needed

              # Add an edge between the stop and the POI with the distance as the weight
              net.add_edge(stop.stop_I, poi, weight=distance)
def _add_stops_to_net(net, stops):
    """
    Add nodes to the network from the pandas dataframe describing (a part of the) stops table in the GTFS database.

    Parameters
    ----------
    net: networkx.Graph
    stops: pandas.DataFrame
    """
    for stop in stops.itertuples():
        data = {
            "lat": stop.lat,
            "lon": stop.lon,
            "name": stop.name
        }
        net.add_node(stop.stop_I, **data)

def temporal_network(gtfs,
                     start_time_ut=None,
                     end_time_ut=None,
                     route_type=None):
    """
    Compute the temporal network of the data, and return it as a pandas.DataFrame

    Parameters
    ----------
    gtfs : gtfspy.GTFS
    start_time_ut: int | None
        start time of the time span (in unix time)
    end_time_ut: int | None
        end time of the time span (in unix time)
    route_type: int | None
        Specifies which mode of public transport are included, or whether all modes should be included.
        The int should be one of the standard GTFS route_types:
        (see also gtfspy.route_types.TRANSIT_ROUTE_TYPES )
        If route_type is not specified, all modes are included.

    Returns
    -------
    events_df: pandas.DataFrame
        Columns: departure_stop, arrival_stop, departure_time_ut, arrival_time_ut, route_type, route_I, trip_I
    """
    events_df = gtfs.get_transit_events(start_time_ut=start_time_ut,
                                        end_time_ut=end_time_ut,
                                        route_type=route_type)
    events_df.drop('to_seq', axis=1, inplace=True)
    events_df.drop('shape_id', axis=1, inplace=True)
    events_df.drop('duration', axis=1, inplace=True)
    events_df.drop('route_id', axis=1, inplace=True)
    events_df.rename(
        columns={
            'from_seq': "seq"
        },
        inplace=True
    )
    return events_df


def route_to_route_network(gtfs, walking_threshold, start_time, end_time):
    """
    Creates networkx graph where the nodes are bus routes and a edge indicates that there is a possibility to transfer
    between the routes
    :param gtfs:
    :param walking_threshold:
    :param start_time:
    :param end_time:
    :return:
    """
    graph = networkx.Graph()
    routes = gtfs.get_table("routes")

    for i in routes.itertuples():
        graph.add_node(i.route_id, type=i.type, color=route_types.ROUTE_TYPE_TO_COLOR[i.type])


    query = """SELECT stop1.route_id AS route_id1, stop1.type, stop2.route_id AS route_id2, stop2.type FROM
                (SELECT * FROM stop_distances WHERE d_walk < %s) sd,
                (SELECT * FROM stop_times, trips, routes 
                WHERE stop_times.trip_I=trips.trip_I AND trips.route_I=routes.route_I 
                AND stop_times.dep_time_ds > %s AND stop_times.dep_time_ds < %s) stop1,
                (SELECT * FROM stop_times, trips, routes 
                WHERE stop_times.trip_I=trips.trip_I AND trips.route_I=routes.route_I 
                AND stop_times.dep_time_ds > %s AND stop_times.dep_time_ds < %s) stop2
                WHERE sd.from_stop_I = stop1.stop_I AND sd.to_stop_I = stop2.stop_I AND stop1.route_id != stop2.route_id
                GROUP BY stop1.route_id, stop2.route_id""" % (walking_threshold, start_time, end_time, start_time,
                                                              end_time)
    df = gtfs.execute_custom_query_pandas(query)

    for items in df.itertuples():
        graph.add_edge(items.route_id1, items.route_id2)
    graph.remove_nodes_from(networkx.isolates(graph))
    return graph




# def cluster_network_stops(stop_to_stop_net, distance):
#     """
#     Aggregate graph by grouping nodes that are within a specified distance.
#     The ids of the nodes are tuples of the original stop_Is.
#
#     Parameters
#     ----------
#     network: networkx.DiGraph
#     distance: float
#         group all nodes within this distance.
#
#     Returns
#     -------
#     graph: networkx.Graph
#     """
#     pass


# def aggregate__network(self, graph, distance):
#     """
#     See to_aggregate_line_graph for documentation
#     """
#     raise NotImplementedError("this is not working fully yet")
#     assert distance <= 1000, "only works with distances below 1000 meters"
#     nodes = set(graph.nodes())
#
#     node_distance_graph = networkx.Graph()
#
#     stop_distances = self.get_table("stop_distances")
#     stop_pairs = stop_distances[stop_distances['d'] <= distance]
#     stop_pairs = zip(stop_pairs['from_stop_I'], stop_pairs['to_stop_I'])
#     for node in nodes:
#         node_distance_graph.add_node(node)
#     for node, another_node in stop_pairs:
#         if (node in nodes) and (another_node in nodes):
#             node_distance_graph.add_edge(node, another_node)
#
#     node_group_iter = networkx.connected_components(node_distance_graph)
#
#     aggregate_graph = networkx.Graph()
#     old_node_to_new_node = {}
#     for node_group in node_group_iter:
#         new_node_id = tuple(node for node in node_group)
#         lats = []
#         lons = []
#         names = []
#         for node in node_group:
#             if node not in graph:
#                 # some stops may not part of the original node line graph
#                 # (e.g. if some lines are not considered, or there are extra stops in stops table)
#                 continue
#             old_node_to_new_node[node] = new_node_id
#             lats.append(graph.node[node]['lat'])
#             lons.append(graph.node[node]['lon'])
#             names.append(graph.node[node]['name'])
#         new_lat = numpy.mean(lats)
#         new_lon = numpy.mean(lons)
#         attr_dict = {
#             "lat": new_lat,
#             "lon": new_lon,
#             "names": names
#         }
#         aggregate_graph.add_node(new_node_id, attr_dict=attr_dict)
#
#     for from_node, to_node, data in graph.edges(data=True):
#         new_from_node = old_node_to_new_node[from_node]
#         new_to_node = old_node_to_new_node[to_node]
#         if aggregate_graph.has_edge(new_from_node, new_to_node):
#             edge_data = aggregate_graph.get_edge_data(new_from_node, new_to_node)
#             edge_data['route_ids'].append(data['route_ids'])
#         else:
#             aggregate_graph.add_edge(new_from_node, new_to_node, route_ids=data['route_ids'])
#     return aggregate_graph
