import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count
from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler

def calculate_travel_times_parallel(cbg_ids, cbg_node_ids, walk_network, transit_events, 
                                    analysis_start_time_ut, analysis_end_time_ut, transfer_margin=120, 
                                    walk_speed=1.5, verbose=False):
    """
    Calculate the average travel time matrix between CBG pairs using parallel processing.
    
    Parameters:
    cbg_ids (list): List of CBG IDs.
    cbg_node_ids (dict): Dictionary mapping CBG IDs to node IDs.
    walk_network (networkx.Graph): Walking network graph.
    transit_events (list): List of transit events (connections).
    analysis_start_time_ut (int): Start time for analysis in UNIX time.
    analysis_end_time_ut (int): End time for analysis in UNIX time.
    transfer_margin (int): Transfer margin in seconds (default is 120).
    walk_speed (float): Walking speed in meters per second (default is 1.5).
    verbose (bool): Whether to print verbose output (default is False).
    
    Returns:
    pd.DataFrame: A DataFrame containing travel times between each CBG pair.
    """
    def process_cbg_origin(origin_cbg_id):
        # Initialize the profiler
        profiler = MultiObjectivePseudoCSAProfiler(
            transit_events=transit_events,
            targets=[cbg_node_ids[cbg_id] for cbg_id in cbg_ids if cbg_id != origin_cbg_id],
            start_time_ut=analysis_start_time_ut,
            end_time_ut=analysis_end_time_ut,
            transfer_margin=transfer_margin,
            walk_network=walk_network,
            walk_speed=walk_speed,
            verbose=verbose,
            track_vehicle_legs=True,
            track_time=True
        )
        
        origin_node_id = cbg_node_ids[origin_cbg_id]
        if origin_node_id not in walk_network.nodes():
            return origin_cbg_id, {}

        # Run the profiler
        profiler.reset([cbg_node_ids[cbg_id] for cbg_id in cbg_ids if cbg_id != origin_cbg_id])
        profiler.run()

        # Extract stop profiles and calculate travel times
        stop_profiles = profiler.stop_profiles
        travel_times = {}
        for target_cbg_id in cbg_ids:
            if target_cbg_id == origin_cbg_id:
                travel_times[target_cbg_id] = 0  # Travel time to itself is zero
            else:
                target_node_id = cbg_node_ids[target_cbg_id]
                profile = stop_profiles.get(target_node_id, None)
                if profile:
                    labels = profile.get_final_optimal_labels()
                    min_time = min([label.arrival_time_target - label.departure_time for label in labels], default=None)
                    travel_times[target_cbg_id] = min_time if min_time else float('nan')
                else:
                    travel_times[target_cbg_id] = float('nan')
        return origin_cbg_id, travel_times

    # Use multiprocessing to process CBGs in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(process_cbg_origin, cbg_ids)

    # Combine results into a DataFrame
    T_cbg = {origin: times for origin, times in results}
    T_cbg_df = pd.DataFrame(T_cbg).T
    T_cbg_df.fillna(9999, inplace=True)  # Replace NaN with a high value to indicate inaccessibility

    return T_cbg_df

# Example usage
# Assuming cbg_ids, cbg_node_ids, walk_network, transit_events, analysis_start_time_ut, and analysis_end_time_ut are defined
# T_cbg_df = calculate_travel_times_parallel(cbg_ids, cbg_node_ids, walk_network, transit_events, analysis_start_time_ut, analysis_end_time_ut)
# T_cbg_df.to_csv('cbg_travel_times_parallel.csv')
