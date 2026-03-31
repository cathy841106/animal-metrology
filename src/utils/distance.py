import math

def euclidean(p1, p2):
    """
    Compute Euclidean distance between two 2D points.
    
    Parameters:
        p1: tuple or list (x, y)
        p2: tuple or list (x, y)
        
    Returns:
        Euclidean distance (float)
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)