import os

from gtfspy import import_gtfs
from gtfspy import gtfs
from gtfspy import osm_transfers


def load_or_import_example_gtfs(verbose=False):
    path='/content/drive/MyDrive/safegraph/google_transit_Birmingham_AL.zip'
    import_gtfs.import_gtfs(path, 'test.db')
    gtfs_conn = sqlite3.connect('test.db')
    GTFS(gtfs_conn)
    
    G = GTFS('test.db')

    if verbose:
        print("Location name:" + G.get_location_name())  # should print Kuopio
        print("Time span of the data in unixtime: " + str(G.get_approximate_schedule_time_span_in_ut()))
        # prints the time span in unix time
    return G


if __name__ == "__main__":
    load_or_import_example_gtfs(verbose=True)
