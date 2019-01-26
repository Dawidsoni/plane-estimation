import rospy
import roslib
import ros_numpy
import time
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import rviz_tools_py as rviz_tools
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Polygon
from tf import transformations

from hough_3d import Hough3D
from geo_transformations import GeoTransformations

#from pointcloud_utils import convert


def pca_refit(plane):
    return plane#TODO
    points = plane[1]
    cov_mtx = np.cov(points)
    w, v = np.linalg.eig(cov_mtx)
    plane_normal = v[:, np.argmin(w)]
    print(plane_normal)
    raise ""


def find_vectors_on_plane(plane_equation):
    vectors = []
    for i in range(3):
        if plane_equation[i] != 0:
            vector = [0, 0, 0]
            vector[i] = -plane_equation[3] / plane_equation[i]
            vectors.append(tuple(vector))
    return vectors[0:2]


def draw_plane(plane_equation):
    v1, v2 = find_vectors_on_plane(plane_equation)
    normal_vector = np.cross(v1, v2)
    vector_norm = np.linalg.norm(normal_vector)
    y_angle = np.arccos(normal_vector[2] / vector_norm)
    y_rotation = transformations.rotation_matrix(y_angle, (0, 1, 0))
    z_angle = np.arctan(plane_equation[1] / plane_equation[0]) if plane_equation[0] != 0 else 0.0
    z_rotation = transformations.rotation_matrix(z_angle, (0, 0, 1))
    distance_from_origin = plane_equation[3] / (vector_norm ** 2)
    translation = transformations.translation_matrix(distance_from_origin * normal_vector)
    transformation = transformations.concatenate_matrices(translation, z_rotation, y_rotation)
    markers.publishPlane(transformation, 5, 5, 'purple', 5.0)


def main(point_cloud):
    t = time.time()
    # Read the point cloud
    points = np.array(tuple(pc2.read_points(point_cloud, field_names=('x', 'y', 'z'), skip_nans=True)))
    #print points.shape
    max_ind = len(points)
    random_points = points[np.random.choice(max_ind, 100)]
    # Perform the Hough transform
    hough = Hough3D(random_points)
    spher_planes = hough.all_possible_planes()
    cart_planes = hough.planes_in_cartesian_normal_form(spher_planes)
    cart_planes = map(pca_refit, cart_planes)
    #votes = hough.voting_phase(cart_planes)
    for i in range(50):
        try:
            draw_plane(cart_planes[i])
        except Exception as exc:
           continue
    return 0#votes


if __name__ == "__main__":
    rospy.init_node('planes')
    rospy.Subscriber('/zed/point_cloud/cloud_filtered', PointCloud2, main, queue_size=1)
    markers = rviz_tools.RvizMarkers('/zed_left_camera_frame', 'visualization_marker')
    rospy.spin()

