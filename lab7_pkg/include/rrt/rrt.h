// RRT assignment
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <tf2_ros/transform_broadcaster.h>

using namespace std;

typedef struct RRT_Node {
    double x, y;
    double cost;
    int parent;
    bool is_root = false;
} RRT_Node;


class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();
private:

    // Publishers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_viz_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_viz_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_viz_pub_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // Random generator
    std::mt19937 gen;
    std::uniform_real_distribution<> x_dist;
    std::uniform_real_distribution<> y_dist;

    // ---- Occupancy grid (car-local frame) ----
    static constexpr int GRID_WIDTH        = 300;
    static constexpr int GRID_HEIGHT       = 300;
    static constexpr double GRID_RESOLUTION = 0.05;
    std::vector<int8_t> occupancy_grid_;

    int  grid_col(double x_local) const;
    int  grid_row(double y_local) const;
    bool is_occupied(double x, double y) const;
    bool is_occupied_wide(double x, double y, double radius) const;
    double wall_distance(double x, double y) const;

    // ---- Tunable parameters ----
    double max_expansion_dist_ = 0.4;
    double goal_threshold_     = 0.5;
    int    max_rrt_iters_      = 2000;
    double lookahead_          = 0.7;
    double search_radius_      = 1.5;
    int    inflate_cells_      = 3;
    double car_half_width_     = 0.30;
    double wall_cost_weight_   = 0.5;

    // ---- Dynamic goal ----
    double goal_x_ = 2.0;
    double goal_y_ = 0.0;

    // ---- Waypoints ----
    std::vector<std::pair<double, double>> waypoints_;
    bool use_waypoints_ = false;
    void load_waypoints(const std::string &filepath);
    void choose_goal_from_waypoints();
    double wp_lookahead_ = 1.5;   // how far ahead on the waypoint path to set the goal

    // ---- Car state ----
    double car_x_ = 0.0, car_y_ = 0.0, car_yaw_ = 0.0;
    bool   scan_received_ = false;

    // Callbacks
    void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg);
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);

    // RRT methods
    std::vector<double> sample();
    int nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point);
    RRT_Node steer(RRT_Node &nearest_node, std::vector<double> &sampled_point);
    bool check_collision(RRT_Node &nearest_node, RRT_Node &new_node);
    bool is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y);
    std::vector<RRT_Node> find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node);

    // RRT* methods
    double cost(std::vector<RRT_Node> &tree, RRT_Node &node);
    double line_cost(RRT_Node &n1, RRT_Node &n2);
    std::vector<int> near(std::vector<RRT_Node> &tree, RRT_Node &node);
};