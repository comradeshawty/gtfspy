import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count
from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler

import multiprocessing
import pandas as pd
import numpy as np

def process_cbg_origin(origin_cbg_id, cbg_node_ids, all_node_ids, connections, start_time, end_time, walk_network):

    origin_node_id = cbg_node_ids[origin_cbg_id]

    # Ensure the origin node is in the set of all nodes
    if origin_node_id not in all_node_ids:
        return origin_cbg_id, {}

    # Prepare targets as other CBG centroid node IDs
    target_node_ids = [cbg_node_ids[cbg_id] for cbg_id in cbg_node_ids if cbg_id != origin_cbg_id]

    # Ensure targets are in the set of all nodes
    valid_targets = [node_id for node_id in target_node_ids if node_id in all_node_ids]

    if not valid_targets:
        return origin_cbg_id, {}

    # Set up the profiler
    mpCSA = MultiObjectivePseudoCSAProfiler(
        connections,
        targets=valid_targets,
        start_time_ut=start_time,
        end_time_ut=end_time,
        transfer_margin=120,  # seconds
        walk_network=walk_network,
        walk_speed=1.5,  # meters per second
        verbose=False,
        track_vehicle_legs=True,
        track_time=True
    )
    mpCSA.run()
    profiles = mpCSA.stop_profiles

    # Extract travel times to each target
    travel_times = {}
    departure_stop_profile = profiles.get(origin_node_id, None)

    if departure_stop_profile:
        labels = departure_stop_profile.get_final_optimal_labels()
        for target_cbg_id in cbg_node_ids:
            if target_cbg_id == origin_cbg_id:
                travel_times[target_cbg_id] = 0  # Travel time to self is zero
                continue
            target_node_id = cbg_node_ids[target_cbg_id]
            min_time = float('inf')
            for label in labels:
                if label.node == target_node_id:
                    travel_time = label.arr_time - label.dep_time
                    if travel_time < min_time:
                        min_time = travel_time
            if np.isfinite(min_time):
                travel_times[target_cbg_id] = min_time
            else:
                travel_times[target_cbg_id] = np.nan
    else:
        # If no profile is available, assign NaN to all targets
        for target_cbg_id in cbg_node_ids:
            if target_cbg_id == origin_cbg_id:
                travel_times[target_cbg_id] = 0
            else:
                travel_times[target_cbg_id] = np.nan

    return origin_cbg_id, travel_times


def calculate_travel_times_parallel(cbg_ids, cbg_node_ids, all_node_ids, connections, start_time, end_time, walk_network):
    # Initialize the travel time matrix as a dictionary
    T_cbg = {}

    # Use multiprocessing Pool to process the travel times in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = [
            pool.apply_async(
                process_cbg_origin,
                (origin_cbg_id, cbg_node_ids, all_node_ids, connections, start_time, end_time, walk_network)
            ) for origin_cbg_id in cbg_ids
        ]

        # Collect the results
        for result in results:
            origin_cbg_id, travel_times = result.get()
            T_cbg[origin_cbg_id] = travel_times

    # Convert the travel time dictionary into a pandas DataFrame
    T_cbg_df = pd.DataFrame(T_cbg).T  # Transpose to have origins as rows and destinations as columns

    # Replace NaN values with a high value to indicate inaccessibility (optional)
    T_cbg_df = T_cbg_df.fillna(9999)

    return T_cbg_df


# Example call
# T_cbg_df = calculate_travel_times_parallel(cbg_ids, cbg_node_ids, all_node_ids, connections, start_time, end_time, walk_network)
# T_cbg_df.to_csv('cbg_travel_times.csv')

