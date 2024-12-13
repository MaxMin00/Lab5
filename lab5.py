import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import euclidean

"""
CS 437 Lab 5: Wildlife Conservation Data Extraction Script
Author: Blake McBride (blakepm2@illinois.edu)

Purpose: To extract data from the READABLE simulation logs into a clean .json format and facilitate analysis
"""

# import necessary libraries
import json
import re
from google.colab import files

# method for parsing only the GPS data at each timestamp for each distinct animal using regex
def parse_gps_data(file_path: str) -> dict:
    """
    Parses GPS data at each timestamp for each distinct animal from a READABLE simulation log file and returns it as a dictionary.

    Args:
        file_path (str): The path to the *READABLE* .txt file containing the GPS data for your animals.

    Returns:
        dict: A dictionary containing all parsed GPS, timestamp, and animal data which can be saved as a clean .json.
    """
    data = {}
    current_animal = None

    # define regex patterns for grabbing animal id, timestamp, and GPS coordinates
    animal_pattern = re.compile(r'^=============== (.*?):(.*?) LOG ===============$')
    timestamp_pattern = re.compile(r'-- Timestamp: ([\d.]+)')
    location_pattern = re.compile(r'"location": \[([-?\d.]+),([-?\d.]+)\]')

    # open and parse the file
    with open(file_path, 'r') as file:
        for line in file:
            # check for animal type and id
            animal_match = animal_pattern.match(line.strip())
            if animal_match:
                animal_type = animal_match.group(1)
                animal_id = animal_match.group(2)
                current_animal = f"{animal_type}:{animal_id}"
                data[current_animal] = {"timestamp": [], "gps coordinates": []}
                continue

            # check for timestamp and GPS coordinates
            if current_animal:
                timestamp_match = timestamp_pattern.match(line.strip())
                location_match = location_pattern.match(line.strip())

                if timestamp_match:
                    current_timestamp = timestamp_match.group(1)
                    data[current_animal]["timestamp"].append(current_timestamp)
                elif location_match:
                    location = (location_match.group(1), location_match.group(2))
                    data[current_animal]["gps coordinates"].append(location)

    return data

def distance_between_two_points(lat1,lon1, lat2,lon2):
    R = 6372800  # Earth radius in meters

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def calculate_speeds(data):
    speeds = defaultdict(list)

    for zebra, info in data.items():
        coordinates = info["gps coordinates"]
        timestamps = info["timestamp"]

        for i in range(1, len(coordinates)):
            current_coord = [float(coordinates[i][0]), float(coordinates[i][1])]
            previous_coord = [float(coordinates[i - 1][0]), float(coordinates[i - 1][1])]

            distance = euclidean(current_coord, previous_coord)
            time_difference = float(timestamps[i]) - float(timestamps[i - 1])

            if time_difference > 0:
                speed = distance / time_difference
                speeds[zebra].append(speed)

    return speeds



# Analyze zebra movement
def analyze_movement(speeds):
    return [speed for zebra_speeds in speeds.values() for speed in zebra_speeds]


# Plot CDF of movement speeds
def plot_speed_cdf(speeds):
    sorted_speeds = np.sort(speeds)
    cdf = np.arange(len(sorted_speeds)) / float(len(sorted_speeds))
    plt.figure()
    plt.plot(sorted_speeds, cdf, marker=".", linestyle="none")
    plt.xlabel("Speed (units per second)")
    plt.ylabel("CDF")
    plt.title("CDF of Zebra Movement Speeds")
    plt.grid()
    plt.show()


def couple_zebra(data, distance_threshold=5):
    zebra_pairs = defaultdict(int)
    zebra_positions = {}
    for zebra, info in data.items():
        coordinates = info["gps coordinates"]
        parsed_coordinates = [list(map(float, coord)) for coord in coordinates]
        zebra_positions[zebra] = parsed_coordinates
        
def couple_zebra(data, distance_threshold=5):
    zebra_pairs = defaultdict(int)

    zebra_positions = {}
    for zebra, info in data.items():
        coordinates = info["gps coordinates"]
        parsed_coordinates = [list(map(float, coord)) for coord in coordinates]
        zebra_positions[zebra] = parsed_coordinates

    for zebra1, positions1 in zebra_positions.items():
        for zebra2, positions2 in zebra_positions.items():
            if zebra1 >= zebra2:
                continue
            for position1, position2 in zip(positions1, positions2):
                distance = euclidean(position1, position2)
                if distance < distance_threshold:
                    zebra_pairs[(zebra1, zebra2)] += 1

    result = {}
    for (zebra1, zebra2), count in zebra_pairs.items():
        total_positions = len(zebra_positions[zebra1])
        if count > total_positions * 0.5:
            result[(zebra1, zebra2)] = count

    return result


# Plot heatmap of time spent
def plot_time_spent_heatmap(data, bin_size=0.5):
    location_counts = defaultdict(int)
    for zebra, info in data.items():
        coords = info["gps coordinates"]
        for coord in coords:
            binned_coord = (
                round(float(coord[0]) / bin_size) * bin_size,
                round(float(coord[1]) / bin_size) * bin_size,
            )
            location_counts[binned_coord] += 1

    x, y, counts = [], [], []
    for (coord_x, coord_y), count in location_counts.items():
        x.append(coord_x)
        y.append(coord_y)
        counts.append(count)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=counts)
    plt.figure(figsize=(12, 10))
    plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="hot",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    plt.colorbar(label="Time Spent (instances)")
    plt.title("Heatmap of Time Spent by Zebras at Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = "parsed_data.json"
    data = load_data(file_path)

    speeds = calculate_speeds(data)
    all_speeds = analyze_movement(speeds)
    couple_z = couple_zebra(data)

    print("Zebra Patterns:", couple_z)

    plot_speed_cdf(all_speeds)
    plot_time_spent_heatmap(data)