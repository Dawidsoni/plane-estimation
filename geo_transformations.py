import numpy as np


class GeoTransformations(object):
    """
    Helpful analytical operations in
    cartesian and spherical coordinates
    """
    
    def spherical_coord_to_cartesian_coord(self, point):
        """
        Convert a point in spherical coordinates
        to a point in euclid coordinates
        """
        
        theta_1, theta_2, r = point
        
        # Cartesian coordinates
        x = np.sin(theta_1) * np.cos(theta_2)
        y = np.sin(theta_1) * np.sin(theta_2)
        z = np.cos(theta_1)
        
        return r * np.array([x, y, z])
    
    def euclid_point_to_plane(self, point, plane):
        """
        Calculate euclid dist between a point
        and a plane, both in euclid coordinates
        
        plane: np.array([a, b, c, d]) such that
        ax + by + cz + d = 0
        
        point: np.array([x, y, z])
        """
        
        numerator = np.abs(np.sum(np.multiply(plane[:-1], point)) + plane[-1])
        denominator = np.sqrt(np.sum(plane[:-1]**2))
        
        return numerator / denominator
    
    def max_dist_point(self, points):
        """
        Find the point with the maximum distance
        from the origin
        """
        
        distances = np.apply_along_axis(lambda point: \
                                        np.sqrt(np.sum(point**2)), 1, points)
        
        return np.max(distances)
