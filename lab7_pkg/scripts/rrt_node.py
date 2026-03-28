#!/usr/bin/env python3
"""
RRT / RRT* local planner for F1TENTH / RoboRacer
Reference: https://arxiv.org/pdf/1105.1186.pdf
"""

import numpy as np
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker


# ─────────────────────────────────────────────────────────────
# Tree node
# ─────────────────────────────────────────────────────────────
class RRTNode(object):
    def __init__(self):
        self.x       = None
        self.y       = None
        self.parent  = None   # index into tree list
        self.cost    = 0.0    # used in RRT*
        self.is_root = False


# ─────────────────────────────────────────────────────────────
# RRT ROS node
# ─────────────────────────────────────────────────────────────
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('pose_topic',       '/odom')
        self.declare_parameter('scan_topic',       '/scan')
        self.declare_parameter('drive_topic',      '/ackermann_cmd')

        # RRT settings
        self.declare_parameter('goal_threshold',   0.3)
        self.declare_parameter('max_expansion',    0.5)
        self.declare_parameter('max_iterations',   400)  # slightly lowered to prevent lag
        self.declare_parameter('goal_bias',        0.15)
        self.declare_parameter('sample_x_min',    -0.5)
        self.declare_parameter('sample_x_max',     6.0)
        self.declare_parameter('sample_y_min',    -3.0)
        self.declare_parameter('sample_y_max',     3.0)

        # Occupancy grid
        self.declare_parameter('grid_resolution',  0.1)   # m / cell
        self.declare_parameter('grid_width',       8.0)   # metres (y-axis)
        self.declare_parameter('grid_height',      12.0)  # metres (x-axis)
        self.declare_parameter('inflation_radius', 0.3)  # metres

        # Pure pursuit
        self.declare_parameter('drive_speed',      1.5)
        self.declare_parameter('lookahead',        0.1)

        # RRT* toggle
        self.declare_parameter('use_rrt_star',     False)
        self.declare_parameter('rrt_star_radius',  1.2)

        p = self.get_parameter
        self.goal_threshold_  = p('goal_threshold').value
        self.max_expansion_   = p('max_expansion').value
        self.max_iterations_  = p('max_iterations').value
        self.goal_bias_       = p('goal_bias').value
        self.sample_x_min_    = p('sample_x_min').value
        self.sample_x_max_    = p('sample_x_max').value
        self.sample_y_min_    = p('sample_y_min').value
        self.sample_y_max_    = p('sample_y_max').value
        self.grid_res_        = p('grid_resolution').value
        self.grid_w_m_        = p('grid_width').value
        self.grid_h_m_        = p('grid_height').value
        self.inflation_r_     = p('inflation_radius').value
        self.drive_speed_     = p('drive_speed').value
        self.lookahead_       = p('lookahead').value
        self.use_rrt_star_    = p('use_rrt_star').value
        self.rrt_star_r_      = p('rrt_star_radius').value

        # Dynamic Goal (Updated by Lidar)
        self.dynamic_goal_x_  = 3.0
        self.dynamic_goal_y_  = 0.0

        # Derived grid dimensions
        self.grid_cols_ = int(self.grid_w_m_ / self.grid_res_)
        self.grid_rows_ = int(self.grid_h_m_ / self.grid_res_)
        self.grid_origin_x_ = -self.grid_h_m_ / 2.0
        self.grid_origin_y_ = -self.grid_w_m_ / 2.0
        self.occupancy_grid_ = np.zeros(
            self.grid_rows_ * self.grid_cols_, dtype=np.int8)

        # ── Subscribers ─────────────────────────────────────────────────────
        pose_topic = p('pose_topic').value
        scan_topic = p('scan_topic').value

        self.pose_sub_ = self.create_subscription(
            Odometry, pose_topic, self.pose_callback, 1)
        self.scan_sub_ = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 1)

        # ── Publishers ──────────────────────────────────────────────────────
        self.drive_pub_ = self.create_publisher(
            AckermannDriveStamped, p('drive_topic').value, 1)
        self.tree_pub_  = self.create_publisher(Marker,        '/rrt_tree', 10)
        self.path_pub_  = self.create_publisher(Marker,        '/rrt_path', 10)
        self.grid_pub_  = self.create_publisher(OccupancyGrid, '/rrt_grid', 10)

        self.get_logger().info(
            f"RRT node started (RRT{'*' if self.use_rrt_star_ else ''})")

    # ─────────────────────────────────────────────────────────
    # scan_callback – build local occupancy grid & set DYNAMIC GOAL
    # ─────────────────────────────────────────────────────────
    def scan_callback(self, scan_msg):
        self.occupancy_grid_[:] = 0

        inflation_cells = int(math.ceil(self.inflation_r_ / self.grid_res_))
        angles = (scan_msg.angle_min
                  + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment)
        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        # 1. Clear out chassis hits (bumped up to 0.35m to clear F1TENTH bumper)
        valid = (np.isfinite(ranges)
                 & (ranges >= max(scan_msg.range_min, 0.35)) 
                 & (ranges <= scan_msg.range_max))
        angles = angles[valid]
        ranges = ranges[valid]

        # 2. DYNAMIC GOAL: Find the longest clear distance in front of the car
        # Look only at the front 180 degrees (-pi/2 to pi/2)
        front_mask = (angles > -1.57) & (angles < 1.57)
        if np.any(front_mask):
            front_ranges = ranges[front_mask]
            front_angles = angles[front_mask]
            max_idx = np.argmax(front_ranges)
            
            # Cap the target distance at 4.0 meters so we don't try to plan too far ahead
            longest_range = min(front_ranges[max_idx], 4.0) 
            self.dynamic_goal_x_ = longest_range * np.cos(front_angles[max_idx])
            self.dynamic_goal_y_ = longest_range * np.sin(front_angles[max_idx])
        else:
            self.dynamic_goal_x_ = 3.0
            self.dynamic_goal_y_ = 0.0

        # Convert to local Cartesian for mapping
        px = ranges * np.cos(angles)
        py = ranges * np.sin(angles)

        # Grid indices
        cols = ((py - self.grid_origin_y_) / self.grid_res_).astype(int)
        rows = ((px - self.grid_origin_x_) / self.grid_res_).astype(int)

        for r, c in zip(rows, cols):
            for dr in range(-inflation_cells, inflation_cells + 1):
                for dc in range(-inflation_cells, inflation_cells + 1):
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < self.grid_rows_ and 0 <= c2 < self.grid_cols_:
                        self.occupancy_grid_[r2 * self.grid_cols_ + c2] = 100

        # Publish grid
        og = OccupancyGrid()
        og.header.stamp    = scan_msg.header.stamp
        og.header.frame_id = 'ego_racecar/base_link'
        og.info.resolution = self.grid_res_
        og.info.width      = self.grid_cols_
        og.info.height     = self.grid_rows_
        og.info.origin.position.x = self.grid_origin_x_
        og.info.origin.position.y = self.grid_origin_y_
        og.data = self.occupancy_grid_.tolist()
        self.grid_pub_.publish(og)

    # ─────────────────────────────────────────────────────────
    # pose_callback – main RRT loop
    # ─────────────────────────────────────────────────────────
    def pose_callback(self, pose_msg):
        tree = []
        root = RRTNode()
        root.x = 0.0; root.y = 0.0
        root.cost = 0.0; root.parent = -1; root.is_root = True
        tree.append(root)

        best_idx = 0
        min_dist = float('inf')

        for _ in range(self.max_iterations_):
            sampled_point = self.sample()
            near_idx = self.nearest(tree, sampled_point)
            new_node = self.steer(tree[near_idx], sampled_point)
            new_node.parent = near_idx

            if not self.check_collision(tree[near_idx], new_node):
                continue

            if self.use_rrt_star_:
                # RRT*
                new_node.cost = (self.cost(tree, tree[near_idx])
                                 + self.line_cost(tree[near_idx], new_node))

                neighbors = self.near(tree, new_node)
                best_parent = near_idx
                best_cost   = new_node.cost
                for nb in neighbors:
                    if self.check_collision(tree[nb], new_node):
                        c = self.cost(tree, tree[nb]) + self.line_cost(tree[nb], new_node)
                        if c < best_cost:
                            best_cost   = c
                            best_parent = nb

                new_node.parent = best_parent
                new_node.cost   = best_cost
                tree.append(new_node)
                new_idx = len(tree) - 1

                for nb in neighbors:
                    if nb == best_parent:
                        continue
                    if self.check_collision(new_node, tree[nb]):
                        c = new_node.cost + self.line_cost(new_node, tree[nb])
                        if c < tree[nb].cost:
                            tree[nb].parent = new_idx
                            tree[nb].cost   = c
            else:
                # Plain RRT
                new_node.cost = (tree[near_idx].cost
                                 + self.line_cost(tree[near_idx], new_node))
                tree.append(new_node)
                new_idx = len(tree) - 1

            # Keep track of the node that gets CLOSEST to the dynamic goal
            dist_to_goal = math.hypot(new_node.x - self.dynamic_goal_x_, 
                                      new_node.y - self.dynamic_goal_y_)
            if dist_to_goal < min_dist:
                min_dist = dist_to_goal
                best_idx = new_idx

            # If it's close enough, we can stop searching early
            if dist_to_goal <= self.goal_threshold_:
                break

        # Always find a path, even if it didn't perfectly hit the goal
        path = self.find_path(tree, best_idx)

        # ── Visualise and Drive ───────────────────────────────────────────────
        stamp = pose_msg.header.stamp
        self._publish_tree(tree, stamp)
        
        # If we successfully built a path with more than just the root node
        if len(path) > 1:
            self._publish_path(path, stamp)
            self._pure_pursuit(path)
        else:
            # Complete failure - BRAKE
            cmd = AckermannDriveStamped()
            cmd.drive.speed          = 0.0
            cmd.drive.steering_angle = 0.0
            self.drive_pub_.publish(cmd)

    # ─────────────────────────────────────────────────────────
    # Helper Functions
    # ─────────────────────────────────────────────────────────
    def sample(self):
        if np.random.rand() < self.goal_bias_:
            return (self.dynamic_goal_x_, self.dynamic_goal_y_)
        x = np.random.uniform(self.sample_x_min_, self.sample_x_max_)
        y = np.random.uniform(self.sample_y_min_, self.sample_y_max_)
        return (x, y)

    def nearest(self, tree, sampled_point):
        dists = [(node.x - sampled_point[0])**2 + (node.y - sampled_point[1])**2
                 for node in tree]
        return int(np.argmin(dists))

    def steer(self, nearest_node, sampled_point):
        dx   = sampled_point[0] - nearest_node.x
        dy   = sampled_point[1] - nearest_node.y
        dist = math.sqrt(dx*dx + dy*dy)

        new_node = RRTNode()
        if dist <= self.max_expansion_:
            new_node.x = sampled_point[0]
            new_node.y = sampled_point[1]
        else:
            ratio      = self.max_expansion_ / dist
            new_node.x = nearest_node.x + ratio * dx
            new_node.y = nearest_node.y + ratio * dy
        return new_node

    def check_collision(self, nearest_node, new_node):
        x0, y0 = nearest_node.x, nearest_node.y
        x1, y1 = new_node.x,     new_node.y

        seg_len = math.sqrt((x1-x0)**2 + (y1-y0)**2)
        steps   = max(int(math.ceil(seg_len / (self.grid_res_ * 0.5))), 1)

        for s in range(steps + 1):
            t  = s / steps
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)

            col = int((py - self.grid_origin_y_) / self.grid_res_)
            row = int((px - self.grid_origin_x_) / self.grid_res_)

            if not (0 <= row < self.grid_rows_ and 0 <= col < self.grid_cols_):
                return False
            if self.occupancy_grid_[row * self.grid_cols_ + col] > 50:
                return False

        return True

    def find_path(self, tree, best_idx):
        path = []
        idx  = best_idx
        while idx >= 0:
            path.append(tree[idx])
            if tree[idx].is_root:
                break
            idx = tree[idx].parent
        path.reverse()
        return path

    def cost(self, tree, node):
        return node.cost

    def line_cost(self, n1, n2):
        dx = n1.x - n2.x
        dy = n1.y - n2.y
        return math.sqrt(dx*dx + dy*dy)

    def near(self, tree, node):
        r2 = self.rrt_star_r_ ** 2
        return [i for i, n in enumerate(tree)
                if (n.x - node.x)**2 + (n.y - node.y)**2 <= r2]

    def _pure_pursuit(self, path):
        """Pure pursuit with Dynamic Lookahead based on speed."""
        
        # 1. Calculate dynamic lookahead (scales with speed)
        # e.g., at 1.5 m/s, lookahead is 0.9m. At 1.0 m/s, lookahead is 0.6m.
        # We clamp it between 0.6m and 1.2m to keep it strictly within safe bounds.
        dynamic_lookahead = max(0.6, min(1.2, self.drive_speed_ * 0.6))

        # 2. Find the target waypoint
        target = path[-1]   # default: goal node
        for node in path:
            d = math.sqrt(node.x**2 + node.y**2)
            if d >= dynamic_lookahead:
                target = node
                break

        # 3. Calculate curvature: κ = 2y / L²
        L2 = target.x**2 + target.y**2
        curvature = (2.0 * target.y / L2) if L2 > 1e-6 else 0.0
        
        # 4. Calculate steering angle
        steer = math.atan(curvature)
        
        # 5. Dynamic Speed Control (slow down for sharp turns!)
        # If the steering angle is steep, drop the speed so we don't understeer
        current_speed = self.drive_speed_
        if abs(steer) > 0.2:  # roughly 11 degrees
            current_speed = self.drive_speed_ * 0.7  # slow down by 30%

        # Clamp steering to physical limits of F1TENTH (±24 degrees or ~0.4189 rad)
        steer = max(-0.4189, min(0.4189, steer))

        # 6. Publish drive command
        cmd = AckermannDriveStamped()
        cmd.drive.speed          = current_speed
        cmd.drive.steering_angle = steer
        self.drive_pub_.publish(cmd)

    def _publish_tree(self, tree, stamp):
        m = Marker()
        m.header.stamp    = stamp
        m.header.frame_id = 'ego_racecar/base_link'
        m.ns              = 'rrt_tree'
        m.id              = 0
        m.type            = Marker.LINE_LIST
        m.action          = Marker.ADD
        m.scale.x         = 0.02
        m.color.r         = 0.0; m.color.g = 0.7
        m.color.b         = 1.0; m.color.a = 0.8

        for node in tree:
            if node.is_root:
                continue
            p1 = Point(); p1.x = node.x;              p1.y = node.y
            p2 = Point(); p2.x = tree[node.parent].x; p2.y = tree[node.parent].y
            m.points.append(p1)
            m.points.append(p2)
        self.tree_pub_.publish(m)

    def _publish_path(self, path, stamp):
        m = Marker()
        m.header.stamp    = stamp
        m.header.frame_id = 'base_link'
        m.ns              = 'rrt_path'
        m.id              = 1
        m.type            = Marker.LINE_STRIP
        m.action          = Marker.ADD
        m.scale.x         = 0.05
        m.color.r         = 1.0; m.color.g = 0.5
        m.color.b         = 0.0; m.color.a = 1.0

        for node in path:
            p = Point(); p.x = node.x; p.y = node.y
            m.points.append(p)
        self.path_pub_.publish(m)


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)
    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()