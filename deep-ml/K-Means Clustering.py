from collections import defaultdict
import math

# 支持任意维度的欧式距离计算
def calculate(points: tuple, centers: tuple) -> float:
    distance = 0.0
    for p, c in zip(points, centers):
        distance += (p - c) ** 2
    return math.sqrt(distance)

def k_means_clustering(
    points: list[tuple[float, ...]],  # 支持任意维度元组
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:
    n = len(points)
    if n == 0 or k == 0:
        return []
    
    # 初始化质心
    centroids = [list(centroid) for centroid in initial_centroids]
    
    for _ in range(max_iterations):
        cluster_assignment = defaultdict(list)
        
        # 1. 分配数据点到最近质心
        for point_idx, point in enumerate(points):
            min_distance = float('inf')
            min_centroid_idx = -1
            for centroid_idx, centroid in enumerate(centroids):
                # 计算欧式距离（三维或更高维）
                dist = calculate(point, centroid)
                if dist < min_distance:
                    min_distance = dist
                    min_centroid_idx = centroid_idx
            cluster_assignment[min_centroid_idx].append(point_idx)
        
        # 2. 更新质心（仅在迭代结束后统一更新）
        new_centroids = []
        for centroid_idx in cluster_assignment:
            indices = cluster_assignment[centroid_idx]
            # 提取该簇所有点的各维度坐标
            coords = list(zip(*[points[i] for i in indices]))
            # 计算各维度均值
            centroid = tuple(sum(dim)/len(dim) for dim in coords)
            new_centroids.append( (centroid_idx, centroid) )
        
        # 处理空簇
        for centroid_idx in range(k):
            if centroid_idx not in [idx for idx, _ in new_centroids]:
                new_centroids.append( (centroid_idx, centroids[centroid_idx]) )
        
        # 按索引排序并更新质心（确保顺序与初始一致）
        new_centroids.sort()
        centroids = [list(centroid) for _, centroid in new_centroids]
    
    # 保留5位小数，转换为元组
    return [ tuple(round(c, 5) for c in centroid) for centroid in centroids ]

if __name__ == "__main__":
    # 二维
    points_2d = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)] 
    k = 2
    initial_centroids_2d = [(1, 1), (10, 1)]
    print(k_means_clustering(points_2d, k, initial_centroids_2d, 10))
    # 输出: [(1.0, 2.0), (10.0, 2.0)]

    # 三维
    points_3d = [(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)]
    initial_centroids_3d = [(1, 1, 1), (10, 10, 10)]
    print(k_means_clustering(points_3d, 2, initial_centroids_3d, 10))
    # 预期输出: [(1.0, 1.0, 1.0), (10.33333, 10.66667, 10.33333)]