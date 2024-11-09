from matplotlib import pyplot as plt
from matplotlib import rc

import example_import
from gtfspy.routing.helpers import get_transit_connections, get_walk_network
from gtfspy.gtfs import GTFS

from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler
from gtfspy.routing.node_profile_analyzer_time_and_veh_legs import NodeProfileAnalyzerTimeAndVehLegs
G = GTFS('test.db')
from collections import defaultdict

# Initialize the matrix to store average transit times
T_transit = defaultdict(dict)

#from_stop_name = "Ahkiotie 2 E"
#to_stop_name = "Kauppahalli P"
#from_stop_I = None
#to_stop_I = None
#stop_dict = G.stops().to_dict("index")
#for stop_I, data in stop_dict.items():
#    if data['name'] == from_stop_name:
 #       from_stop_I = stop_I
 #   if data['name'] == to_stop_name:
 #       to_stop_I = stop_I
#assert (from_stop_I is not None)
#assert (to_stop_I is not None)

# The start and end times between which PT operations (and footpaths) are scanned:
ANALYSIS_START_TIME_UT = G.get_suitable_date_for_daily_extract(ut=True) + 10 * 3600 
# Analyze tremporal distances / travel times for one hour departure time interval:
ANALYSIS_END_TIME_UT = ANALYSIS_START_TIME_UT + 1 * 3600
# Normally scanning of PT connections (i.e. "routing") should start at the same time of the analysis start time:
CONNECTION_SCAN_START_TIME_UT = ANALYSIS_START_TIME_UT
# Consider only journey alternatives that arrive to the destination at most two hours
# later than last departure time of interest:
CONNECTION_SCAN_END_TIME_UT = ANALYSIS_END_TIME_UT + 2 * 3600

connections = get_transit_connections(G, CONNECTION_SCAN_START_TIME_UT, CONNECTION_SCAN_END_TIME_UT)
MAX_WALK_LENGTH = 1000
# Get the walking network with all stop-pairs that are less than 1000 meters apart.
walk_network = get_walk_network(G, MAX_WALK_LENGTH)
# Note that if you would want enable longer walking distances, e.g. 2000 meters, then you
# (may) need to recompute the footpath lengths between stops with
# gtfspy.osm_transfers.add_walk_distances_to_db_python(..., cutoff_distance_m=2000).

# Iterate over all pairs of CBGs
for i, n_o in enumerate(cbg_ids):
    T_transit[n_o] = {}
    origin_stops = cbg_to_stops[n_o]
    
    for n_d in cbg_ids:
        destination_stops = cbg_to_stops[n_d]
        travel_times = []
        
        # Compute travel times between all combinations of origin and destination stops
        for from_stop_I in origin_stops:
            for to_stop_I in destination_stops:
                # Convert stop IDs to integers if necessary
                from_stop_I = int(from_stop_I)
                to_stop_I = int(to_stop_I)
                
                # Set up the profiler
                mpCSA = MultiObjectivePseudoCSAProfiler(connections,
                                                        targets=[to_stop_I],
                                                        start_time_ut=ANALYSIS_START_TIME_UT,
                                                        end_time_ut=ANALYSIS_END_TIME_UT,
                                                        transfer_margin=120,
                                                        walk_network=walk_network,
                                                        walk_speed=1.5,
                                                        verbose=False,
                                                        track_vehicle_legs=True,
                                                        track_time=True)
                mpCSA.run()
                profiles = mpCSA.stop_profiles
                
                departure_stop_profile = profiles.get(from_stop_I, None)
                if departure_stop_profile:
                    analyzer = NodeProfileAnalyzerTimeAndVehLegs(departure_stop_profile.get_final_optimal_labels(),
                                                                 float('inf'),
                                                                 ANALYSIS_START_TIME_UT,
                                                                 ANALYSIS_END_TIME_UT)
                    mean_time = analyzer.mean_temporal_distance()
                    if np.isfinite(mean_time):
                        travel_times.append(mean_time)
        
        # Compute average travel time between n_o and n_d
        if travel_times:
            T_transit[n_o][n_d] = np.mean(travel_times)
        else:
            T_transit[n_o][n_d] = np.nan  # Unreachable or no valid paths
    print(f"Processed CBG {i+1}/{len(cbg_ids)}")


# stop_dict = G.stops().to_dict("index")
#print("Origin: ", stop_dict[from_stop_I])
#print("Destination: ", stop_dict[to_stop_I])
#print("Minimum temporal distance: ", analyzer.min_temporal_distance() / 60., " minutes")
#print("Mean temporal distance: ", analyzer.mean_temporal_distance() / 60., " minutes")
#print("Medan temporal distance: ", analyzer.median_temporal_distance() / 60., " minutes")
#print("Maximum temporal distance: ", analyzer.max_temporal_distance() / 60., " minutes")
# Note that the mean and max temporal distances have the value of `direct_walk_duration`,
# if there are no journey alternatives departing after (or at the same time as) `ANALYSIS_END_TIME_UT`.
# Thus, if you obtain a float('inf') value for some of the temporal distance measures, it could probably be
# avoided by increasing the value of PT_CONNECTIONS_SCANNING_END_TIME_UT (while taking care that
# The median temporal distance is often more robust to this kind of service level variations.


#timezone_pytz = G.get_timezone_pytz()
#print("Plotting...")

# use tex in plotting
#rc("text", usetex=True)
#fig1 = analyzer.plot_new_transfer_temporal_distance_profile(timezone=timezone_pytz,
                                                            format_string="%H:%M")
#fig2 = analyzer.plot_temporal_distance_pdf_horizontal(use_minutes=True)
#print("Showing...")
#plt.show()
