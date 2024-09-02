import math
from time import perf_counter

def angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def distance(p1, p2):
    return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2

def ccw(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def graham_scan(points):
    # Find the lowest point
    start = min(points, key=lambda p: (p[1], p[0]))
    points.pop(points.index(start))

    # Sort points by polar angle with start
    points.sort(key=lambda p: (angle(start, p), -distance(start, p)))

    # Initialize stack
    hull = [start]
    for p in points:
        while len(hull) > 1 and ccw(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull

def main():
    points = [(1, 1), (2, 5), (3, 3), (5, 3), (3, 2), (2, 2), (7, 2), (1, 5), (2, 5)]
    t0 = perf_counter()
    convex_hull = graham_scan(points)
    t1 = perf_counter()
    print("TIMING", t1 - t0)
    print(convex_hull)

if __name__ == '__main__':
    main()

# Example usage
