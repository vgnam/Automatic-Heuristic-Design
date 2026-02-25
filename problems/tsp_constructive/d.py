import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from collections import Counter

# Tải dataset
data = np.load(r'dataset/train50_dataset.npy', allow_pickle=True)
print(data.shape)
print(f"Dataset shape: {len(data)} instances")

# Kiểm tra một instance mẫu
print(f"First instance: {data[0].shape} points")


def extract_features(points):
    """Trích xuất đặc trưng từ tọa độ điểm"""
    points = np.array(points)
    n = len(points)

    if n < 2:
        return {}

    # Khoảng cách giữa các điểm
    distances = pdist(points)
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)

    # Bounding box
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = max(width, height) / (min(width, height) + 1e-8)

    # Mật độ và độ "cụm"
    area = width * height
    density = n / (area + 1e-8)
    cluster_score = std_dist / (avg_dist + 1e-8)  # cao = nhiều cụm

    # Độ đồng đều (uniformity)
    min_dist = np.min(distances[distances > 1e-6])
    max_dist = np.max(distances)
    uniformity = min_dist / (max_dist + 1e-8)

    return {
        'n_points': n,
        'avg_dist': avg_dist,
        'std_dist': std_dist,
        'aspect_ratio': aspect_ratio,
        'density': density,
        'cluster_score': cluster_score,
        'uniformity': uniformity,
        'width': width,
        'height': height
    }


def classify_instance(features):
    """
    Gán nhãn mô tả cho instance dựa trên đặc trưng
    """
    n = features['n_points']
    cluster_score = features['cluster_score']
    aspect_ratio = features['aspect_ratio']
    uniformity = features['uniformity']
    width, height = features['width'], features['height']

    # 1. CỤM (Clustered)
    if cluster_score > 0.7:
        return "clustered"

    # 2. DÀI - HẸP (Elongated)
    elif aspect_ratio > 3:
        if width > height:
            return "horizontal_path"
        else:
            return "vertical_path"

    # 3. PHÂN BỐ ĐỀU (Uniform)
    elif uniformity > 0.6 and cluster_score < 0.5:
        return "uniform"

    # 4. HÌNH TRÒN (Circular)
    elif 0.8 < aspect_ratio < 1.2 and cluster_score < 0.5:
        return "circular"

    # 5. LƯỚI (Grid-like)
    elif uniformity > 0.5 and abs(width - height) < 0.3 * max(width, height):
        return "grid"

    # 6. PHỨC TẠP (Complex)
    else:
        return "complex"


# Phân tích tất cả instances
all_features = []
all_labels = []

for i, instance in enumerate(data):
    features = extract_features(instance)
    label = classify_instance(features)

    all_features.append(features)
    all_labels.append(label)

    print(f"Instance {i:2d}: {label:12s} | "
          f"Points: {features['n_points']:2d} | "
          f"Cluster: {features['cluster_score']:.2f} | "
          f"Aspect: {features['aspect_ratio']:.2f}")

# Thống kê
label_counts = Counter(all_labels)
print("\nThống kê nhãn:")
for label, count in label_counts.items():
    print(f"  {label}: {count} instances")

