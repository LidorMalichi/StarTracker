import math

import numpy as np
import pandas as pd
from itertools import combinations
from StarTriangleHash import create_key_val  # Import from your current script


def get_affine_transform(p_img, p_db):
    """
    Estimate affine transform from 3 image points to 3 database points
    """
    A = np.hstack([p_img, np.ones((3, 1))])
    B = p_db
    transform, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return transform.T  # Returns 2x3 matrix


def apply_affine_transform(stars, transform):
    """
    Apply affine transformation to a list of (x, y) coordinates
    """
    stars_hom = np.hstack([stars, np.ones((stars.shape[0], 1))])
    return np.dot(transform, stars_hom.T).T


def find_matches(transformed_stars, db_stars, tolerance=5):
    """
    Match transformed stars to DB stars within given pixel tolerance.
    Returns list of (image_star_index, db_star_index)
    """
    matches = []
    for i, (tx, ty) in enumerate(transformed_stars):
        for j, row in db_stars.iterrows():
            dx, dy = row['x'], row['y']
            if np.linalg.norm([tx - dx, ty - dy]) <= tolerance:
                matches.append((i, row['#']))  # return indices, not positions
                break  # Avoid duplicate matches
    return matches


def match_stars(image_csv, hash_table, db_csv, max_distance, min_distance, tolerance=5):
    """
    Match stars from image using hash table and affine transform.

    Args:
        image_csv: CSV file path of stars from image (x, y)
        hash_table: Precomputed triangle angle hash table
        db_csv: CSV file path of database stars (x, y, #)
        tolerance: Maximum pixel distance for a match

    Returns:
        DataFrame of matched (x_image, y_image, x_db, y_db)
    """
    image_stars = pd.read_csv(image_csv)
    db_stars = pd.read_csv(db_csv)

    image_coords = image_stars[['x', 'y']].values
    image_ids = image_stars['#'].values  # Use index if '#' not present

    for i, j, k in combinations(range(len(image_coords)), 3):
        p1, p2, p3 = image_coords[i], image_coords[j], image_coords[k]
        s1, s2, s3 = image_ids[i], image_ids[j], image_ids[k]

        # Calculate distances between all pairs for filtering
        dist1 = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)  # distance between s1 and s2
        dist2 = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)  # distance between s2 and s3
        dist3 = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)  # distance between s3 and s1

        # Skip if triangle exceeds maximum distance
        if max_distance is not None and max(dist1, dist2, dist3) > max_distance:
            continue

        if min_distance is not None and min(dist1, dist2, dist3) < min_distance:
            continue

        key, _ = create_key_val(p1, p2, p3, s1, s2, s3)

        if key not in hash_table:
            continue

        for val in hash_table[key]:
            ids = list(map(int, val.split("_")))
            db_subset = db_stars[db_stars['#'].isin(ids)]

            if len(db_subset) != 3:
                continue

            db_points = db_subset[['x', 'y']].values
            transform = get_affine_transform(np.array([p1, p2, p3]), db_points)
            transformed_coords = apply_affine_transform(image_coords, transform)

            matches = find_matches(transformed_coords, db_stars, tolerance)

            if len(matches) >= len(image_coords)*0.6:
                results = []
                for img_idx, db_idx in matches:
                    x_img, y_img = image_stars.iloc[img_idx][['x', 'y']]
                    id_img = image_stars.iloc[img_idx]['#'] if '#' in image_stars.columns else img_idx

                    x_db, y_db = db_stars.iloc[db_idx][['x', 'y']]
                    id_db = db_stars.iloc[db_idx]['#'] if '#' in db_stars.columns else db_idx

                    results.append([id_img, x_img, y_img, id_db, x_db, y_db])

                matched_df = pd.DataFrame(results, columns=['id_image', 'x_image', 'y_image', 'id_db', 'x_db', 'y_db'])
                print(f"Found {len(results)} matches using triplet {s1}, {s2}, {s3}")
                return matched_df

    print("No match found.")
    return pd.DataFrame(columns=['x_image', 'y_image', 'x_db', 'y_db'])


def load_hash_table(filename):
    """
    Load a hash table from file
    """
    hash_table = {}
    with open(filename, 'r') as f:
        for line in f:
            key, val = line.strip().split(",")
            if key not in hash_table:
                hash_table[key] = []
            hash_table[key].append(val)
    return hash_table


if __name__ == "__main__":
    hash_table = load_hash_table("triangle_angle_hash.csv")
    matches = match_stars("stars.csv", hash_table, "starsdb.csv", max_distance=3000, min_distance=0, tolerance=20)

    for _, row in matches.iterrows():
        print(f"Image star (ID: {row['id_image']}) at ({row['x_image']:.2f}, {row['y_image']:.2f}) "
              f"matches DB star (ID: {row['id_db']}) at ({row['x_db']:.2f}, {row['y_db']:.2f})")

