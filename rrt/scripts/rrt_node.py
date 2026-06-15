#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ============================================================================
# PARAMETERS
# ============================================================================
STEER_POLARITY = 1.0  
WAYPOINTS_FILE = "/home/mshrekhar/roboracer_ws/src/lab-6-motion-planning-team8/lab7_pkg/waypoints.csv"

REFERENCE_SPEED = 1.5   
HORIZON = 35            
N_SAMPLES = 400         
DT = 0.05
WHEELBASE = 0.33
MAX_STEER = 0.41 
LAMBDA = 0.15           

W_GOAL   = 150.0  
W_TRACK  = 25.0   
W_WALL   = 400.0  
W_SMOOTH = 50.0   

# ============================================================================
class MPPITrackAligned(Node):
    def __init__(self):
        super().__init__("mppi_track_aligned")
        
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
        
        self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_cb, qos)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, qos)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.viz_pub = self.create_publisher(MarkerArray, "/mppi_trajectories", 10)

        self.U = np.zeros(HORIZON)
        self.last_scan_pts = None
        
        # Robust CSV Loading
        try:
            self.waypoints = np.loadtxt(WAYPOINTS_FILE, delimiter=',')
            if self.waypoints.shape[1] > 3: # Handle CSVs with more than 3 columns
                self.waypoints = self.waypoints[:, :3]
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"CSV Error: {e}")
            exit()

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        mask = (ranges > 0.1) & (ranges < 4.0)
        self.last_scan_pts = np.column_stack((ranges[mask] * np.cos(angles[mask]), ranges[mask] * np.sin(angles[mask])))

    def odom_cb(self, msg):
        if self.last_scan_pts is None: return

        # 1. POSITION
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        curr_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        # 2. BETTER WAYPOINT SELECTION
        # Step A: Find the absolute closest waypoint to the car
        dists = np.sqrt((self.waypoints[:, 0] - curr_x)**2 + (self.waypoints[:, 1] - curr_y)**2)
        closest_idx = np.argmin(dists)
        
        # Step B: Look ahead from the closest index (approx 2.0m ahead)
        # Assuming waypoints are ~0.1m apart, 20 indices ahead is 2m.
        target_idx = (closest_idx + 20) % len(self.waypoints)
        target_wp = self.waypoints[target_idx]

        # 3. TRANSFORM (POLAR)
        dx, dy = target_wp[0] - curr_x, target_wp[1] - curr_y
        global_angle = math.atan2(dy, dx)
        rel_angle = global_angle - curr_yaw
        while rel_angle > np.pi: rel_angle -= 2*np.pi
        while rel_angle < -np.pi: rel_angle += 2*np.pi

        # Local Goal
        dist_to_wp = math.sqrt(dx**2 + dy**2)
        gx = dist_to_wp * math.cos(rel_angle)
        gy = dist_to_wp * math.sin(rel_angle)

        # DEBUG LOG
        if self.get_clock().now().nanoseconds % 1000000000 < 100000000:
            side = "LEFT" if rel_angle > 0 else "RIGHT"
            self.get_logger().info(f"Target is {dist_to_wp:.2f}m away, {abs(math.degrees(rel_angle)):.1f} deg to the {side}")

        # 4. MPPI
        noise = np.random.normal(0.0, 0.15, (N_SAMPLES, HORIZON))
        V = np.clip(self.U + noise, -MAX_STEER, MAX_STEER)
        states = np.zeros((N_SAMPLES, 3)) 
        costs = np.zeros(N_SAMPLES)
        traj_viz = np.zeros((N_SAMPLES, HORIZON, 2))

        for t in range(HORIZON):
            steer = V[:, t]
            states[:, 0] += REFERENCE_SPEED * np.cos(states[:, 2]) * DT
            states[:, 1] += REFERENCE_SPEED * np.sin(states[:, 2]) * DT
            states[:, 2] += STEER_POLARITY * (REFERENCE_SPEED / WHEELBASE) * np.tan(steer) * DT
            
            traj_viz[:, t, 0] = states[:, 0]
            traj_viz[:, t, 1] = states[:, 1]

            alpha = (t + 1) / HORIZON
            costs += W_TRACK * ((states[:, 0] - alpha*gx)**2 + (states[:, 1] - alpha*gy)**2)

            for pt in self.last_scan_pts[::50]:
                d2 = (states[:, 0] - pt[0])**2 + (states[:, 1] - pt[1])**2
                costs += W_WALL * np.exp(-d2 / 0.25)

        costs += W_GOAL * ((states[:, 0] - gx)**2 + (states[:, 1] - gy)**2)
        costs += W_SMOOTH * np.sum(np.square(np.diff(V, axis=1)), axis=1)

        # 5. UPDATE
        weights = np.exp(-(costs - np.min(costs)) / LAMBDA)
        self.U = np.sum(weights[:, None] * V, axis=0) / (np.sum(weights) + 1e-9)

        # 6. DRIVE
        msg_out = AckermannDriveStamped()
        msg_out.drive.steering_angle = float(self.U[0])
        msg_out.drive.speed = REFERENCE_SPEED * (1.0 - 0.5 * abs(self.U[0]/MAX_STEER))
        self.drive_pub.publish(msg_out)

        self.publish_trajectories(traj_viz)
        self.U = np.roll(self.U, -1); self.U[-1] = 0.0

    def publish_trajectories(self, traj_buffer):
        marker_array = MarkerArray()
        for i in range(0, N_SAMPLES, 20):
            m = Marker()
            m.header.frame_id = "ego_racecar/base_link"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.02
            m.color.a, m.color.g, m.color.b = 0.5, 1.0, 1.0
            m.points.append(Point(x=0.0, y=0.0, z=0.0))
            for t in range(HORIZON):
                m.points.append(Point(x=traj_buffer[i, t, 0], y=traj_buffer[i, t, 1], z=0.0))
            marker_array.markers.append(m)
        self.viz_pub.publish(marker_array)

def main():
    rclpy.init(); rclpy.spin(MPPITrackAligned()); rclpy.shutdown()

if __name__ == "__main__":
    main()