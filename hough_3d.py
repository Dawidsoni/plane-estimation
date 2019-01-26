import numpy as np
from geo_transformations import GeoTransformations


class Hough3D(GeoTransformations):
    
    def __init__(self, point_cloud, dist_to_plane=0.05,
                 threshold=10, r_diff=1.,
                 theta_1=np.pi / 10, theta_2=np.pi / 10):
        self.points = point_cloud
        self.dist = dist_to_plane
        self.threshold = threshold
        self.r_diff = r_diff
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        
        # Limit the searching to a sphere
        self.r_of_space = self.max_dist_point(self.points)

    def all_possible_planes(self):
        """
        Find all possible planes, each coresponding
        to the unique (r, theta1, theta2) tuple
        """

        # TODO check the angle axis
        theta_1_coord = np.arange(0., 2*np.pi + self.theta_1, self.theta_1)
        theta_2_coord = np.hstack((np.arange(-1/2 * np.pi,
                                             1/2 * np.pi + self.theta_2,
                                             self.theta_2),
                                   np.arange(3/4 * np.pi,
                                             5/4 * np.pi + self.theta_2,
                                             self.theta_2)))

        # Each pair of angles represents a family of parallel planes
        planes = np.dstack(np.meshgrid(theta_1_coord,
                                       theta_2_coord)).reshape(-1, 2)

        # For each pair append the first r value to obtain the first
        # plane in each family, it defines all the planes, as we know
        # all the translations differs by const
        first_r_repeat = np.repeat(1, planes.shape[0])[:, None]
        planes = np.append(planes, first_r_repeat, axis=1)

        return planes

    def plane_in_cartesian_normal_form(self, spher_coord):
        """
        Transform a single triple of spherical coords
        into quadruple (A, B, C, D) in cartesian coords
        """

        normal_vec = self.spherical_coord_to_cartesian_coord(spher_coord)

        # Knowing that the normal vec to plane begins at O,
        # the plane formula is A(x - A) + B(y - B) + C(z - C) = 0
        d = -np.sum(normal_vec**2)
        cart_coord = np.hstack((normal_vec, d))

        # Normalize
        cart_coord /= np.sqrt(np.sum(cart_coord[:3]**2))

        return cart_coord
    
    def planes_in_cartesian_normal_form(self, planes):
        """
        Transform points in spherical coordinates
        to quadruples (A, B, C, D), each coresponding
        to a unique plane which represent the family
        of parallel planes
        """

        cart_planes = np.apply_along_axis(self.plane_in_cartesian_normal_form,
                                          1, planes)       

        return cart_planes

    def voting_phase(self, cart_planes):
        """
        Create an array of votes for each plane
        in the cartesian normal form
        """

        # Possible transitions
        transitions = np.arange(np.ceil(self.r_of_space) + 1)

        # Compute ax + by + cz for each pair (plane, point)
        ax_by_cz = np.dot(cart_planes[:, :3], self.points.T)

        found_planes = []

        for ind, plane_family in enumerate(ax_by_cz):

            # Round each point to the closer plane
            votes_in_family, _ = np.histogram(plane_family, bins=transitions)
            # print votes_in_family

            # Filter the transitions which have ge votes than the self.threshold
            found_transitions = np.ravel(np.argwhere(votes_in_family >= self.threshold))
            # print found_transitions

            # Create the found planes if there is at least one in the family
            if found_transitions.size:
                a_b_c = cart_planes[ind, :3]

                a_b_c_repeat = np.tile(a_b_c, (found_transitions.size, 1))

                transitions = found_transitions + 0.5
                
                found_planes.append(np.hstack((a_b_c_repeat, transitions[:, None])))

        # print found_planes

        return found_planes
        