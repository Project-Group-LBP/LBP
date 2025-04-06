import numpy as np
from PIL import Image


def decode_grayscale_image(image_path, map_width=16, wall_width=4, threshold=0.1):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img) / 255.0

    y_indices, x_indices = np.where(img_array > threshold)

    # Group adjacent bright pixels into data points
    potential_points = []
    for i in range(len(y_indices)):
        x, y = x_indices[i], y_indices[i]
        potential_points.append((x, y))

    # Cluster adjacent pixels
    data_point_clusters = []
    processed = set()

    for point in potential_points:
        if point in processed:
            continue

        # Start a new cluster
        cluster = [point]
        processed.add(point)

        # Process queue for connected pixels
        queue = [point]
        while queue:
            px, py = queue.pop(0)

            # Check 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor = (px + dx, py + dy)
                    if neighbor in potential_points and neighbor not in processed:
                        cluster.append(neighbor)
                        processed.add(neighbor)
                        queue.append(neighbor)

        # Only consider clusters that are likely data points (not too large)
        if 1 <= len(cluster) <= 8:  # Data point is 2x2
            data_point_clusters.append(cluster)

    # Calculate the center of each cluster
    data_points = []
    for cluster in data_point_clusters:
        x_coords = [p[0] for p in cluster]
        y_coords = [p[1] for p in cluster]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        data_points.append((center_x, center_y))

    # Convert from pixel coordinates to logical coordinates
    points = []
    for px, py in data_points:
        # Reverse the transformation from the env class
        # Original transform: x, y = int(4 * x + wall_width * 2), int(4 * y + wall_width * 2)
        logical_x = (px - wall_width * 2) / 4.0
        logical_y = (py - wall_width * 2) / 4.0

        # Only include points that are within the logical grid and divide by width
        if 0 <= logical_x < map_width and 0 <= logical_y < map_width:
            points.append((logical_x / map_width, logical_y / map_width))

    return points


def export(data_points, output_path):
    # Convert to a proper list representation if it's a numpy array
    if isinstance(data_points, np.ndarray):
        # Format as a list of lists with proper commas
        formatted_data = str([[float(x) for x in point] for point in data_points]).replace("], [", "],\n [")
    else:
        # If it's already a list of tuples or lists
        formatted_data = str(data_points).replace("), (", "),\n (")
    with open(output_path, "w") as f:
        f.write("import numpy as np\n")
        f.write(f"decoded_data = {formatted_data}\n")
    print(f"Data points exported to {output_path}")


def input_image(image_path, is_export=True):
    export_path = "decoded_points.py"  # Replace with your export path
    data_points = decode_grayscale_image(image_path)
    if is_export:
        export(data_points, export_path)
    return data_points


def random_points_generator(num_points, is_export=True):
    poi_data = np.random.random(size=(num_points, 2))
    export_path = "decoded_points.py"  # Replace with your export path
    if is_export:
        export(poi_data, export_path)
    return poi_data


def uniform_points_generator(map_width=16, is_export=True):
    poi_data = np.array([[x, y] for x in range(0, map_width, 1) for y in range(0, map_width, 1)]) / map_width
    export_path = "decoded_points.py"  # Replace with your export path
    if is_export:
        export(poi_data, export_path)
    return poi_data
