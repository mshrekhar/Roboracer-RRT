// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
//
// ---- LAB 6 REQUIREMENTS CHECKLIST ----
// [x] Occupancy grid (local frame, ray-traced, inflated, published for RVIZ/Foxglove)
// [x] RRT: sample, nearest, steer, check_collision, is_goal, find_path
// [x] RRT*: cost, line_cost, near, rewiring (extra credit)
// [x] Collision checking with car width (corridor check)
// [x] Trajectory execution via Pure Pursuit (built-in, no separate node needed)
// [x] Visualization: tree (/rrt_tree), path (/rrt_path), goal (/rrt_goal), grid (/occupancy_grid)
// [x] Wall-distance cost in RRT* to prefer corridor-center paths
// [x] Goal-biased sampling (15% bias toward goal)
// [x] Waypoint-based goal selection (from pure pursuit CSV)
// [x] Handles straights and turns

#include "rrt/rrt.h"

// ============================================================================
// Destructor
// ============================================================================
RRT::~RRT() {
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// ============================================================================
// Constructor
// ============================================================================
RRT::RRT()
    : rclcpp::Node("rrt_node"),
      gen((std::random_device())()),
      x_dist(0.5, 4.0),
      y_dist(-2.5, 2.5)
{
    // Publishers
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
        "/drive", 1);
    occ_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/occupancy_grid", 1);
    tree_viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/rrt_tree", 1);
    path_viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/rrt_path", 1);
    goal_viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/rrt_goal", 1);

    // Subscribers
    string pose_topic = "ego_racecar/odom";
    string scan_topic = "/scan";
    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        pose_topic, 1,
        std::bind(&RRT::pose_callback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic, 1,
        std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    // Init grid
    occupancy_grid_.resize(GRID_WIDTH * GRID_HEIGHT, 0);

    // ---- Load waypoints from your pure pursuit CSV ----
    load_waypoints("/home/milan/roboracer_ws/src/lab-6-motion-planning-team8/lab7_pkg/waypoints.csv");

    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Created new RRT Object.");
}

// ============================================================================
// load_waypoints — reads x,y per line from CSV
// ============================================================================
void RRT::load_waypoints(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        RCLCPP_WARN(rclcpp::get_logger("RRT"),
            "Could not open waypoint file: %s — falling back to gap-finding", filepath.c_str());
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // skip comments/empty
        std::stringstream ss(line);
        double x, y;
        char comma;
        if (ss >> x >> comma >> y) {
            waypoints_.push_back({x, y});
        }
    }
    if (!waypoints_.empty()) {
        use_waypoints_ = true;
        RCLCPP_INFO(rclcpp::get_logger("RRT"), "Loaded %zu waypoints", waypoints_.size());
    }
}

// ============================================================================
// choose_goal_from_waypoints — pick a waypoint ahead of the car as the RRT goal
//
// This is the approach from the lecture slides (slide 60):
//   "Choose a local goal from reference path (similar to pure pursuit)"
//
// 1. Find the nearest waypoint to the car
// 2. Walk forward along the waypoint list by wp_lookahead_ distance
// 3. Transform that waypoint into the car's local frame
// 4. That becomes the RRT goal
// ============================================================================
void RRT::choose_goal_from_waypoints() {
    if (waypoints_.empty()) return;

    // Step 1: Find nearest waypoint
    double min_dist = std::numeric_limits<double>::max();
    int nearest_idx = 0;
    for (size_t i = 0; i < waypoints_.size(); ++i) {
        double dx = waypoints_[i].first - car_x_;
        double dy = waypoints_[i].second - car_y_;
        double d = dx * dx + dy * dy;
        if (d < min_dist) {
            min_dist = d;
            nearest_idx = static_cast<int>(i);
        }
    }

    // Step 2: Walk forward along waypoints by wp_lookahead_ distance
    double accum_dist = 0.0;
    int goal_idx = nearest_idx;
    for (size_t steps = 0; steps < waypoints_.size(); ++steps) {
        int curr = (nearest_idx + steps) % waypoints_.size();
        int next = (nearest_idx + steps + 1) % waypoints_.size();
        double dx = waypoints_[next].first - waypoints_[curr].first;
        double dy = waypoints_[next].second - waypoints_[curr].second;
        accum_dist += std::sqrt(dx * dx + dy * dy);
        if (accum_dist >= wp_lookahead_) {
            goal_idx = next;
            break;
        }
    }

    // Step 3: Transform global waypoint → car-local frame
    double dx = waypoints_[goal_idx].first - car_x_;
    double dy = waypoints_[goal_idx].second - car_y_;
    double cos_yaw = std::cos(-car_yaw_);
    double sin_yaw = std::sin(-car_yaw_);
    goal_x_ = dx * cos_yaw - dy * sin_yaw;
    goal_y_ = dx * sin_yaw + dy * cos_yaw;

    // Safety: if the goal ended up behind the car, push it forward
    if (goal_x_ < 0.3) {
        goal_x_ = 1.5;
        goal_y_ = 0.0;
    }

    // Cap goal distance to stay within sampling region (max 3.5m)
    double goal_dist = std::sqrt(goal_x_ * goal_x_ + goal_y_ * goal_y_);
    if (goal_dist > 3.5) {
        goal_x_ = goal_x_ / goal_dist * 3.5;
        goal_y_ = goal_y_ / goal_dist * 3.5;
    }
}

// ============================================================================
// Grid helpers
// ============================================================================
int RRT::grid_col(double x_local) const {
    return static_cast<int>(x_local / GRID_RESOLUTION);
}

int RRT::grid_row(double y_local) const {
    return static_cast<int>(y_local / GRID_RESOLUTION) + GRID_HEIGHT / 2;
}

bool RRT::is_occupied(double x, double y) const {
    int c = grid_col(x);
    int r = grid_row(y);
    if (c < 0 || c >= GRID_WIDTH || r < 0 || r >= GRID_HEIGHT)
        return true;
    return occupancy_grid_[r * GRID_WIDTH + c] > 50;
}

bool RRT::is_occupied_wide(double x, double y, double radius) const {
    int cells = static_cast<int>(std::ceil(radius / GRID_RESOLUTION));
    int cx = grid_col(x);
    int cy = grid_row(y);
    double r2 = radius * radius;
    for (int dc = -cells; dc <= cells; ++dc) {
        for (int dr = -cells; dr <= cells; ++dr) {
            double real_dx = dc * GRID_RESOLUTION;
            double real_dy = dr * GRID_RESOLUTION;
            if (real_dx * real_dx + real_dy * real_dy > r2) continue;
            int nc = cx + dc;
            int nr = cy + dr;
            if (nc < 0 || nc >= GRID_WIDTH || nr < 0 || nr >= GRID_HEIGHT)
                return true;
            if (occupancy_grid_[nr * GRID_WIDTH + nc] > 50)
                return true;
        }
    }
    return false;
}

double RRT::wall_distance(double x, double y) const {
    int cx = grid_col(x);
    int cy = grid_row(y);
    int max_search = 20;
    for (int radius = 1; radius <= max_search; ++radius) {
        for (int dc = -radius; dc <= radius; ++dc) {
            for (int dr = -radius; dr <= radius; ++dr) {
                if (std::abs(dc) != radius && std::abs(dr) != radius) continue;
                int nc = cx + dc;
                int nr = cy + dr;
                if (nc < 0 || nc >= GRID_WIDTH || nr < 0 || nr >= GRID_HEIGHT)
                    return radius * GRID_RESOLUTION;
                if (occupancy_grid_[nr * GRID_WIDTH + nc] > 50)
                    return radius * GRID_RESOLUTION;
            }
        }
    }
    return max_search * GRID_RESOLUTION;
}

// ============================================================================
// scan_callback — build occupancy grid
//
// If using waypoints: goal is set in pose_callback, NOT here.
// If NOT using waypoints: falls back to gap-finding goal (corridor center).
// ============================================================================
void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // ---------- 1. Occupancy grid ----------
    // Start FREE — unseen areas (like around corners) are passable.
    // Only laser endpoints are marked occupied. This lets the RRT tree
    // grow toward waypoints the laser can't see yet.
    std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), 0);

    double angle = scan_msg->angle_min;
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
        double r = scan_msg->ranges[i];
        if (std::isnan(r) || std::isinf(r) ||
            r < scan_msg->range_min || r > scan_msg->range_max) {
            angle += scan_msg->angle_increment;
            continue;
        }

        // Mark endpoint + circular inflation as occupied
        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);
        int gc = grid_col(r * cos_a);
        int gr = grid_row(r * sin_a);
        for (int dc = -inflate_cells_; dc <= inflate_cells_; ++dc) {
            for (int dr = -inflate_cells_; dr <= inflate_cells_; ++dr) {
                if (dc * dc + dr * dr > inflate_cells_ * inflate_cells_) continue;
                int nc = gc + dc;
                int nr = gr + dr;
                if (nc >= 0 && nc < GRID_WIDTH && nr >= 0 && nr < GRID_HEIGHT)
                    occupancy_grid_[nr * GRID_WIDTH + nc] = 100;
            }
        }
        angle += scan_msg->angle_increment;
    }

    // ---------- 2. Fallback gap-finding goal (only if no waypoints) ----------
    if (!use_waypoints_) {
        int num_beams = static_cast<int>(scan_msg->ranges.size());
        double best_score = -1.0;
        int best_idx = num_beams / 2;
        int start_beam = num_beams / 4;
        int end_beam   = 3 * num_beams / 4;

        for (int i = start_beam; i < end_beam; ++i) {
            double r = scan_msg->ranges[i];
            if (std::isnan(r) || std::isinf(r) || r < 0.5) continue;
            double a = scan_msg->angle_min + i * scan_msg->angle_increment;
            double sample_dist = std::min(r * 0.7, 2.0);
            double px = sample_dist * std::cos(a);
            double py = sample_dist * std::sin(a);

            double perp_x = -std::sin(a);
            double perp_y =  std::cos(a);
            double left_dist = 0.0;
            for (double d = 0.05; d < 2.0; d += 0.05) {
                if (is_occupied(px + perp_x * d, py + perp_y * d)) break;
                left_dist = d;
            }
            double right_dist = 0.0;
            for (double d = 0.05; d < 2.0; d += 0.05) {
                if (is_occupied(px - perp_x * d, py - perp_y * d)) break;
                right_dist = d;
            }

            double corridor_width = left_dist + right_dist;
            double min_side = std::min(left_dist, right_dist);
            if (min_side < car_half_width_ + 0.10) continue;

            double center_offset = std::abs(i - num_beams / 2.0) / (num_beams / 4.0);
            double center_bias = 1.0 - 0.3 * center_offset;
            double score = corridor_width * std::min(r, 3.0) * center_bias;

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        double best_angle = scan_msg->angle_min + best_idx * scan_msg->angle_increment;
        double br = scan_msg->ranges[best_idx];
        if (std::isnan(br) || std::isinf(br)) br = 2.0;
        double sd = std::min(br * 0.6, 2.5);
        if (sd < 1.0) sd = 1.0;

        double px = sd * std::cos(best_angle);
        double py = sd * std::sin(best_angle);
        double perp_x = -std::sin(best_angle);
        double perp_y =  std::cos(best_angle);
        double ld = 0.0, rd = 0.0;
        for (double d = 0.05; d < 2.0; d += 0.05) {
            if (is_occupied(px + perp_x * d, py + perp_y * d)) break;
            ld = d;
        }
        for (double d = 0.05; d < 2.0; d += 0.05) {
            if (is_occupied(px - perp_x * d, py - perp_y * d)) break;
            rd = d;
        }
        double shift = (ld - rd) / 2.0;
        goal_x_ = px - perp_x * shift;
        goal_y_ = py - perp_y * shift;

        double goal_angle = std::atan2(goal_y_, goal_x_);
        if (std::abs(goal_angle) > 1.0) {
            double clamped = std::clamp(goal_angle, -1.0, 1.0);
            double dist = std::sqrt(goal_x_ * goal_x_ + goal_y_ * goal_y_);
            goal_x_ = dist * std::cos(clamped);
            goal_y_ = dist * std::sin(clamped);
        }
    }

    scan_received_ = true;

    // ---------- 3. Publish occupancy grid ----------
    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header.stamp    = this->now();
    grid_msg.header.frame_id = "ego_racecar/base_link";
    grid_msg.info.resolution = GRID_RESOLUTION;
    grid_msg.info.width      = GRID_WIDTH;
    grid_msg.info.height     = GRID_HEIGHT;
    grid_msg.info.origin.position.x = 0.0;
    grid_msg.info.origin.position.y = -(GRID_HEIGHT / 2) * GRID_RESOLUTION;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.data = occupancy_grid_;
    occ_grid_pub_->publish(grid_msg);
}

// ============================================================================
// pose_callback — RRT MAIN LOOP
// ============================================================================
void RRT::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    if (!scan_received_) return;

    // Extract car pose
    car_x_ = pose_msg->pose.pose.position.x;
    car_y_ = pose_msg->pose.pose.position.y;
    double qx = pose_msg->pose.pose.orientation.x;
    double qy = pose_msg->pose.pose.orientation.y;
    double qz = pose_msg->pose.pose.orientation.z;
    double qw = pose_msg->pose.pose.orientation.w;
    car_yaw_ = std::atan2(2.0 * (qw * qz + qx * qy),
                           1.0 - 2.0 * (qy * qy + qz * qz));

    // ---- Set goal from waypoints (if loaded) ----
    if (use_waypoints_) {
        choose_goal_from_waypoints();
    }

    RCLCPP_INFO_THROTTLE(rclcpp::get_logger("RRT"), *this->get_clock(), 1000,
        "Car: (%.1f, %.1f, yaw=%.2f) Goal local: (%.2f, %.2f) Waypoints: %s",
        car_x_, car_y_, car_yaw_, goal_x_, goal_y_,
        use_waypoints_ ? "YES" : "NO (gap-finding)");

    // ---- Build tree ----
    std::vector<RRT_Node> tree;
    RRT_Node root;
    root.x = 0.0; root.y = 0.0;
    root.cost = 0.0; root.parent = -1; root.is_root = true;
    tree.push_back(root);

    double gx = goal_x_;
    double gy = goal_y_;

    // Publish goal marker
    {
        visualization_msgs::msg::Marker gm;
        gm.header.frame_id = "ego_racecar/base_link";
        gm.header.stamp = this->now();
        gm.ns = "goal"; gm.id = 0;
        gm.type = visualization_msgs::msg::Marker::SPHERE;
        gm.action = visualization_msgs::msg::Marker::ADD;
        gm.pose.position.x = gx; gm.pose.position.y = gy;
        gm.scale.x = 0.3; gm.scale.y = 0.3; gm.scale.z = 0.3;
        gm.color.b = 1.0; gm.color.a = 1.0;
        goal_viz_pub_->publish(gm);
    }

    bool path_found    = false;
    int  goal_node_idx = -1;
    std::uniform_real_distribution<> bias_dist(0.0, 1.0);

    // ---- RRT main loop ----
    for (int iter = 0; iter < max_rrt_iters_; ++iter) {
        // 1) Sample — 15% goal bias
        std::vector<double> sampled(2);
        if (bias_dist(gen) < 0.15) {
            sampled[0] = gx;
            sampled[1] = gy;
        } else {
            sampled = sample();
        }

        // 2) Nearest
        int nearest_idx = nearest(tree, sampled);

        // 3) Steer
        RRT_Node new_node = steer(tree[nearest_idx], sampled);

        // 4) Collision check (car-width corridor)
        if (check_collision(tree[nearest_idx], new_node))
            continue;

        // Reject node too close to walls
        if (is_occupied_wide(new_node.x, new_node.y, car_half_width_))
            continue;

        new_node.parent = nearest_idx;
        new_node.cost   = tree[nearest_idx].cost
                        + line_cost(tree[nearest_idx], new_node);

        // ---- RRT* rewiring ----
        std::vector<int> neighbours = near(tree, new_node);
        for (int ni : neighbours) {
            double tentative = tree[ni].cost + line_cost(tree[ni], new_node);
            if (tentative < new_node.cost) {
                if (!check_collision(tree[ni], new_node)) {
                    new_node.parent = ni;
                    new_node.cost   = tentative;
                }
            }
        }
        tree.push_back(new_node);
        int new_idx = static_cast<int>(tree.size()) - 1;
        for (int ni : neighbours) {
            double tentative = new_node.cost + line_cost(new_node, tree[ni]);
            if (tentative < tree[ni].cost) {
                if (!check_collision(new_node, tree[ni])) {
                    tree[ni].parent = new_idx;
                    tree[ni].cost   = tentative;
                }
            }
        }

        // 5) Goal check
        if (is_goal(new_node, gx, gy)) {
            path_found    = true;
            goal_node_idx = new_idx;
            break;
        }
    }

    // ---- Visualise tree ----
    {
        visualization_msgs::msg::MarkerArray tree_markers;
        visualization_msgs::msg::Marker edges;
        edges.header.frame_id = "ego_racecar/base_link";
        edges.header.stamp = this->now();
        edges.ns = "tree"; edges.id = 0;
        edges.type = visualization_msgs::msg::Marker::LINE_LIST;
        edges.action = visualization_msgs::msg::Marker::ADD;
        edges.scale.x = 0.01;
        edges.color.g = 1.0; edges.color.a = 0.4;
        for (size_t i = 1; i < tree.size(); ++i) {
            if (tree[i].parent < 0) continue;
            geometry_msgs::msg::Point p1, p2;
            p1.x = tree[tree[i].parent].x; p1.y = tree[tree[i].parent].y;
            p2.x = tree[i].x; p2.y = tree[i].y;
            edges.points.push_back(p1);
            edges.points.push_back(p2);
        }
        tree_markers.markers.push_back(edges);
        tree_viz_pub_->publish(tree_markers);
    }

    // ---- Fallback if no path found ----
    if (!path_found) {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("RRT"), *this->get_clock(), 500,
            "No path found — tree has %zu nodes, goal at (%.2f, %.2f), goal occupied: %s",
            tree.size(), gx, gy,
            is_occupied(gx, gy) ? "YES" : "no");
        double angle_to_goal = std::atan2(gy, gx);
        double steering = std::clamp(angle_to_goal, -0.4189, 0.4189);
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.drive.steering_angle = steering;
        drive_msg.drive.speed          = 0.3;
        drive_pub_->publish(drive_msg);
        return;
    }

    // ---- Extract & visualise path ----
    std::vector<RRT_Node> path = find_path(tree, tree[goal_node_idx]);
    {
        visualization_msgs::msg::Marker pm;
        pm.header.frame_id = "ego_racecar/base_link";
        pm.header.stamp = this->now();
        pm.ns = "path"; pm.id = 0;
        pm.type = visualization_msgs::msg::Marker::LINE_STRIP;
        pm.action = visualization_msgs::msg::Marker::ADD;
        pm.scale.x = 0.05;
        pm.color.r = 1.0; pm.color.a = 1.0;
        for (auto &n : path) {
            geometry_msgs::msg::Point p;
            p.x = n.x; p.y = n.y; p.z = 0.0;
            pm.points.push_back(p);
        }
        path_viz_pub_->publish(pm);
    }

    // ---- Pure Pursuit on the RRT path ----
    double tx = path.back().x;
    double ty = path.back().y;
    for (auto &n : path) {
        double d = std::sqrt(n.x * n.x + n.y * n.y);
        if (d >= lookahead_) {
            tx = n.x; ty = n.y;
            break;
        }
    }

    double L = std::sqrt(tx * tx + ty * ty);
    if (L < 0.001) L = 0.001;
    double curvature = 2.0 * ty / (L * L);
    double steering  = std::atan(curvature * 0.3302);
    steering = std::clamp(steering, -0.4189, 0.4189);

    double speed = 3.0;
    if (std::abs(steering) > 0.10) speed = 2.5;
    if (std::abs(steering) > 0.25) speed = 2.0;

    auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
    drive_msg.drive.steering_angle = steering;
    drive_msg.drive.speed          = speed;
    drive_pub_->publish(drive_msg);
}

// ============================================================================
// sample
// ============================================================================
std::vector<double> RRT::sample() {
    std::vector<double> sampled_point(2);
    for (int attempt = 0; attempt < 100; ++attempt) {
        double sx = x_dist(gen);
        double sy = y_dist(gen);
        if (!is_occupied_wide(sx, sy, car_half_width_)) {
            sampled_point[0] = sx;
            sampled_point[1] = sy;
            return sampled_point;
        }
    }
    sampled_point[0] = x_dist(gen);
    sampled_point[1] = y_dist(gen);
    return sampled_point;
}

// ============================================================================
// nearest
// ============================================================================
int RRT::nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point) {
    int nearest_node = 0;
    double min_dist  = std::numeric_limits<double>::max();
    for (size_t i = 0; i < tree.size(); ++i) {
        double dx = tree[i].x - sampled_point[0];
        double dy = tree[i].y - sampled_point[1];
        double d2 = dx * dx + dy * dy;
        if (d2 < min_dist) {
            min_dist     = d2;
            nearest_node = static_cast<int>(i);
        }
    }
    return nearest_node;
}

// ============================================================================
// steer
// ============================================================================
RRT_Node RRT::steer(RRT_Node &nearest_node, std::vector<double> &sampled_point) {
    RRT_Node new_node;
    double dx   = sampled_point[0] - nearest_node.x;
    double dy   = sampled_point[1] - nearest_node.y;
    double dist = std::sqrt(dx * dx + dy * dy);

    if (dist <= max_expansion_dist_) {
        new_node.x = sampled_point[0];
        new_node.y = sampled_point[1];
    } else {
        new_node.x = nearest_node.x + (dx / dist) * max_expansion_dist_;
        new_node.y = nearest_node.y + (dy / dist) * max_expansion_dist_;
    }
    new_node.cost    = 0.0;
    new_node.parent  = -1;
    new_node.is_root = false;
    return new_node;
}

// ============================================================================
// check_collision — car-width corridor check
//   Returns TRUE if collision
// ============================================================================
bool RRT::check_collision(RRT_Node &nearest_node, RRT_Node &new_node) {
    double dx   = new_node.x - nearest_node.x;
    double dy   = new_node.y - nearest_node.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 1e-6) return is_occupied_wide(new_node.x, new_node.y, car_half_width_);

    double ux = dx / dist;
    double uy = dy / dist;
    double nx = -uy;
    double ny =  ux;

    double step      = GRID_RESOLUTION * 0.5;
    int    num_steps = static_cast<int>(std::ceil(dist / step));

    for (int i = 0; i <= num_steps; ++i) {
        double t  = (num_steps == 0) ? 0.0 : static_cast<double>(i) / num_steps;
        double cx = nearest_node.x + t * dx;
        double cy = nearest_node.y + t * dy;

        if (is_occupied(cx, cy)) return true;
        if (is_occupied(cx + nx * car_half_width_, cy + ny * car_half_width_)) return true;
        if (is_occupied(cx - nx * car_half_width_, cy - ny * car_half_width_)) return true;
    }
    return false;
}

// ============================================================================
// is_goal
// ============================================================================
bool RRT::is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y) {
    double dx = latest_added_node.x - goal_x;
    double dy = latest_added_node.y - goal_y;
    return std::sqrt(dx * dx + dy * dy) < goal_threshold_;
}

// ============================================================================
// find_path
// ============================================================================
std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree,
                                      RRT_Node &latest_added_node) {
    std::vector<RRT_Node> found_path;
    int current = -1;
    for (size_t i = 0; i < tree.size(); ++i) {
        if (&tree[i] == &latest_added_node) {
            current = static_cast<int>(i);
            break;
        }
    }
    if (current < 0) return found_path;
    while (current != -1) {
        found_path.push_back(tree[current]);
        current = tree[current].parent;
    }
    std::reverse(found_path.begin(), found_path.end());
    return found_path;
}

// ============================================================================
// RRT* methods
// ============================================================================
double RRT::cost(std::vector<RRT_Node> &tree, RRT_Node &node) {
    double c = 0.0;
    int idx = -1;
    for (size_t i = 0; i < tree.size(); ++i) {
        if (&tree[i] == &node) { idx = static_cast<int>(i); break; }
    }
    if (idx < 0) return 0.0;
    int current = idx;
    while (tree[current].parent != -1) {
        int p = tree[current].parent;
        c += line_cost(tree[p], tree[current]);
        current = p;
    }
    return c;
}

double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
    double dx = n1.x - n2.x;
    double dy = n1.y - n2.y;
    double euclidean = std::sqrt(dx * dx + dy * dy);

    double mx = (n1.x + n2.x) / 2.0;
    double my = (n1.y + n2.y) / 2.0;
    double wd = wall_distance(mx, my);

    double max_wd = 1.0;
    double clamped_wd = std::min(wd, max_wd);
    double wall_penalty = wall_cost_weight_ * euclidean * (1.0 - clamped_wd / max_wd);

    return euclidean + wall_penalty;
}

std::vector<int> RRT::near(std::vector<RRT_Node> &tree, RRT_Node &node) {
    std::vector<int> neighborhood;
    double r2 = search_radius_ * search_radius_;
    for (size_t i = 0; i < tree.size(); ++i) {
        double dx = tree[i].x - node.x;
        double dy = tree[i].y - node.y;
        if (dx * dx + dy * dy <= r2)
            neighborhood.push_back(static_cast<int>(i));
    }
    return neighborhood;
}