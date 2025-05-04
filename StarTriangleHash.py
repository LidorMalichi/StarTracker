import numpy as np
import pandas as pd
from itertools import combinations
import math

def calculate_angle(p1, p2, p3):
    """
    Calculate angle between three points with p2 as the vertex

    Args:
        p1: First point as (x, y) coordinates
        p2: Vertex point as (x, y) coordinates
        p3: Third point as (x, y) coordinates

    Returns:
        Angle in radians
    """
    # Vectors from vertex to the other points
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Compute dot product and clip to valid range for arccos
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    # Return angle
    return math.degrees(math.acos(dot_product))


def round_to_range(value, precision=1):
    """
    Round value to specified precision

    Args:
        value: Value to round
        precision: Number of decimal places

    Returns:
        Rounded value as string
    """
    return str(round(value))


def create_key_val(p1, p2, p3, s1, s2, s3):
    # Calculate angles at each vertex
    angle1 = calculate_angle(p3, p1, p2)  # angle at vertex s1
    angle2 = calculate_angle(p1, p2, p3)  # angle at vertex s2
    angle3 = calculate_angle(p2, p3, p1)  # angle at vertex s3

    # Filter out triangles with any angle ≤ 20 degrees
    if angle1 <= 20 or angle2 <= 20 or angle3 <= 20:
        return None, None

    # Create key and value
    key = str(round_to_range(angle1)) + "_" + str(round_to_range(angle2)) + "_" + str(round_to_range(angle3))
    value = str(s1) + "_" + str(s2) + "_" + str(s3)

    return key, value

def create_triangle_angle_hash(stars_df, max_distance=None, min_distance=None):
    """
    Create a triangle hash from star database using angles

    Args:
        stars_df: DataFrame with star data (columns: x, y, #)
        max_distance: Maximum distance between stars to form triangles (optional)

    Returns:
        hash_table: Dictionary with triangle angle hashes as keys and star IDs as values
    """
    # Initialize hash table
    hash_table = {}

    # Extract coordinates and IDs
    coords = []
    star_ids = []

    for idx, star in stars_df.iterrows():
        coords.append((star['x'], star['y']))
        star_ids.append(star['#'])

    print(f"Building triangle angle hash from {len(coords)} stars...")

    # Generate all possible triangles
    count = 0
    for i, j, k in combinations(range(len(coords)), 3):
        p1, p2, p3 = coords[i], coords[j], coords[k]
        s1, s2, s3 = star_ids[i], star_ids[j], star_ids[k]

        # Calculate distances between all pairs for filtering
        dist1 = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)  # distance between s1 and s2
        dist2 = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)  # distance between s2 and s3
        dist3 = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)  # distance between s3 and s1

        # Skip if triangle exceeds maximum distance
        if max_distance is not None and max(dist1, dist2, dist3) > max_distance:
            continue

        if min_distance is not None and min(dist1, dist2, dist3) < min_distance:
            continue

        key, value = create_key_val(p1, p2, p3, s1, s2, s3)

        # Store in hash table
        if  key is not None:
            if key not in hash_table:
                hash_table[key] = []
            hash_table[key].append(value)

        count += 1
        if count % 100000 == 0:
            print(f"Processed {count} triangles...")

    print(f"Built hash table with {len(hash_table)} unique triangle patterns")
    print(f"Total triangles: {count}")

    return hash_table


def save_hash_table(hash_table, filename):
    """
    Save hash table to a file

    Args:
        hash_table: Dictionary with triangle pattern hashes
        filename: Name of output file
    """
    with open(filename, 'w') as file:
        for key, values in hash_table.items():
            # Each value could be a list of multiple star triangle patterns
            for value in values:
                file.write(f"{key},{value}\n")

    print(f"Hash table saved to {filename}")


def main():
    """
    Main function to demonstrate usage
    """
    # Load star database
    stars_db = pd.read_csv('starsdb.csv')

    # Create triangle hash using angles
    hash_table = create_triangle_angle_hash(stars_db, max_distance=2000, min_distance=250)

    # Save hash table
    save_hash_table(hash_table, 'triangle_angle_hash.csv')

    # Print some statistics
    total_patterns = sum(len(values) for values in hash_table.values())
    print(f"Angle hash table contains {total_patterns} triangle patterns")
    print(f"Number of unique angle keys: {len(hash_table)}")


if __name__ == "__main__":
    main()