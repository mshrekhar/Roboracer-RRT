#include "rrt/rrt.h"
#include <queue>

RRT::~RRT() {
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "RRT shutting down");
}

RRT::RRT()
    : rclcpp::Node("rrt_node"),
      gen((std::random_device())()),
      x_dist(-0.5, 5.0),
      y_dist(-3.5, 3.5)
{
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/ackermann_cmd", 1);
    occ_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_grid", 1);
    tree_viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/rrt_tree", 1);
    path_viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_path", 1);
    goal_viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_goal", 1);

    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/pf/pose/odom", 1, std::bind(&RRT::pose_callback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 1, std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    occupancy_grid_.resize(GRID_WIDTH * GRID_HEIGHT, 0);
    distance_grid_.resize(GRID_WIDTH * GRID_HEIGHT, 100.0f);

    load_waypoints("/home/nvidia/f1tenth_ws/src/lab-6-motion-planning-team8/lab7_pkg/waypoints.csv");
    RCLCPP_INFO(this->get_logger(), "RRT Node Ready (fast mode, no rewiring).");
}

// ============================================================================
// Waypoints
// ============================================================================

void RRT::load_waypoints(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        RCLCPP_WARN(this->get_logger(), "Could not open waypoint file: %s", filepath.c_str());
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double x, y; char comma;
        if (ss >> x >> comma >> y) waypoints_.push_back({x, y});
    }
    if (!waypoints_.empty()) {
        use_waypoints_ = true;
        RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_.size());
    }
}

void RRT::choose_goal_from_waypoints() {
    if (waypoints_.empty()) return;

    double min_dist = std::numeric_limits<double>::max();
    int nearest_idx = 0;
    for (size_t i = 0; i < waypoints_.size(); ++i) {
        double dx = waypoints_[i].first - car_x_, dy = waypoints_[i].second - car_y_;
        double d = dx * dx + dy * dy;
        if (d < min_dist) { min_dist = d; nearest_idx = static_cast<int>(i); }
    }

    int best_goal_idx = -1;
    double best_goal_x = 1.5, best_goal_y = 0.0;

    for (double la = wp_lookahead_; la <= wp_lookahead_ * 4.0; la += 0.5) {
        double accum = 0.0;
        int goal_idx = nearest_idx;
        for (size_t s = 0; s < waypoints_.size(); ++s) {
            int curr = (nearest_idx + s) % waypoints_.size();
            int next = (nearest_idx + s + 1) % waypoints_.size();
            double ddx = waypoints_[next].first - waypoints_[curr].first;
            double ddy = waypoints_[next].second - waypoints_[curr].second;
            accum += std::sqrt(ddx * ddx + ddy * ddy);
            if (accum >= la) { goal_idx = next; break; }
        }

        double dx = waypoints_[goal_idx].first - car_x_;
        double dy = waypoints_[goal_idx].second - car_y_;
        double cos_yaw = std::cos(-car_yaw_), sin_yaw = std::sin(-car_yaw_);
        double lx = dx * cos_yaw - dy * sin_yaw;
        double ly = dx * sin_yaw + dy * cos_yaw;
        if (lx < 0.3) continue;
        double gd = std::sqrt(lx * lx + ly * ly);
        if (gd > 3.5) { lx = lx / gd * 3.5; ly = ly / gd * 3.5; }

        if (!is_occupied_wide(lx, ly, car_half_width_)) {
            best_goal_x = lx; best_goal_y = ly; best_goal_idx = goal_idx;
            bool clear = true;
            for (double t = 0.3; t < 0.8; t += 0.2)
                if (is_occupied_wide(lx * t, ly * t, car_half_width_)) { clear = false; break; }
            if (clear) break;
        }
    }

    if (best_goal_idx < 0) {
        double accum = 0.0;
        int goal_idx = nearest_idx;
        for (size_t s = 0; s < waypoints_.size(); ++s) {
            int curr = (nearest_idx + s) % waypoints_.size();
            int next = (nearest_idx + s + 1) % waypoints_.size();
            double ddx = waypoints_[next].first - waypoints_[curr].first;
            double ddy = waypoints_[next].second - waypoints_[curr].second;
            accum += std::sqrt(ddx * ddx + ddy * ddy);
            if (accum >= wp_lookahead_) { goal_idx = next; break; }
        }
        double dx = waypoints_[goal_idx].first - car_x_;
        double dy = waypoints_[goal_idx].second - car_y_;
        double cos_yaw = std::cos(-car_yaw_), sin_yaw = std::sin(-car_yaw_);
        best_goal_x = dx * cos_yaw - dy * sin_yaw;
        best_goal_y = dx * sin_yaw + dy * cos_yaw;
        if (best_goal_x < 0.3) { best_goal_x = 1.5; best_goal_y = 0.0; }
        double gd = std::sqrt(best_goal_x * best_goal_x + best_goal_y * best_goal_y);
        if (gd > 3.5) { best_goal_x /= gd / 3.5; best_goal_y /= gd / 3.5; }
    }

    goal_x_ = best_goal_x;
    goal_y_ = best_goal_y;
}

// ============================================================================
// Grid helpers
// ============================================================================

int RRT::grid_col(double x_local) const { return static_cast<int>(x_local / GRID_RESOLUTION); }
int RRT::grid_row(double y_local) const { return static_cast<int>(y_local / GRID_RESOLUTION) + GRID_HEIGHT / 2; }

bool RRT::is_occupied(double x, double y) const {
    int c = grid_col(x), r = grid_row(y);
    if (c < 0 || c >= GRID_WIDTH || r < 0 || r >= GRID_HEIGHT) return true;
    return occupancy_grid_[r * GRID_WIDTH + c] > OCC_THRESHOLD;
}

bool RRT::is_occupied_wide(double x, double y, double radius) const {
    return wall_distance(x, y) < radius;
}

double RRT::wall_distance(double x, double y) const {
    int c = grid_col(x), r = grid_row(y);
    if (c < 0 || c >= GRID_WIDTH || r < 0 || r >= GRID_HEIGHT) return 0.0;
    return static_cast<double>(distance_grid_[r * GRID_WIDTH + c]);
}

// ============================================================================
// Scan callback
// ============================================================================

void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), static_cast<int8_t>(0));
    std::fill(distance_grid_.begin(), distance_grid_.end(), 100.0f);

    double angle = scan_msg->angle_min;
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
        double r = scan_msg->ranges[i];
        if (std::isnan(r) || std::isinf(r) || r < 0.35) { angle += scan_msg->angle_increment; continue; }

        double r_clamped = std::min(r, static_cast<double>(GRID_WIDTH) * GRID_RESOLUTION * 0.9);
        int gc = grid_col(r_clamped * std::cos(angle));
        int gr = grid_row(r_clamped * std::sin(angle));
        if (gc >= 0 && gc < GRID_WIDTH && gr >= 0 && gr < GRID_HEIGHT)
            occupancy_grid_[gr * GRID_WIDTH + gc] = 100;

        angle += scan_msg->angle_increment;
    }

    // Light inflation
    if (inflate_cells_ > 0) {
        std::vector<int8_t> raw = occupancy_grid_;
        int r2 = inflate_cells_ * inflate_cells_;
        for (int r = 0; r < GRID_HEIGHT; ++r) {
            for (int c = 0; c < GRID_WIDTH; ++c) {
                if (raw[r * GRID_WIDTH + c] <= OCC_THRESHOLD) continue;
                for (int dr = -inflate_cells_; dr <= inflate_cells_; ++dr) {
                    for (int dc = -inflate_cells_; dc <= inflate_cells_; ++dc) {
                        if (dr * dr + dc * dc > r2) continue;
                        int nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < GRID_HEIGHT && nc >= 0 && nc < GRID_WIDTH)
                            occupancy_grid_[nr * GRID_WIDTH + nc] = 100;
                    }
                }
            }
        }
    }

    // BFS distance transform
    std::queue<std::pair<int, int>> q;
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; ++i) {
        if (occupancy_grid_[i] > OCC_THRESHOLD) {
            distance_grid_[i] = 0.0f;
            q.push({i / GRID_WIDTH, i % GRID_WIDTH});
        }
    }
    int ddr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int ddc[] = {0, 0, -1, 1, -1, 1, -1, 1};
    while (!q.empty()) {
        auto [cr, cc] = q.front(); q.pop();
        float d = distance_grid_[cr * GRID_WIDTH + cc];
        for (int i = 0; i < 8; i++) {
            int nr = cr + ddr[i], nc = cc + ddc[i];
            if (nr >= 0 && nr < GRID_HEIGHT && nc >= 0 && nc < GRID_WIDTH) {
                float di = (i < 4) ? (float)GRID_RESOLUTION : (float)GRID_RESOLUTION * 1.414f;
                if (distance_grid_[nr * GRID_WIDTH + nc] > d + di) {
                    distance_grid_[nr * GRID_WIDTH + nc] = d + di;
                    q.push({nr, nc});
                }
            }
        }
    }

    scan_received_ = true;

    nav_msgs::msg::OccupancyGrid gm;
    gm.header.stamp = this->now(); gm.header.frame_id = "base_link";
    gm.info.resolution = GRID_RESOLUTION; gm.info.width = GRID_WIDTH; gm.info.height = GRID_HEIGHT;
    gm.info.origin.position.y = -(GRID_HEIGHT / 2) * GRID_RESOLUTION;
    gm.data = occupancy_grid_;
    occ_grid_pub_->publish(gm);
}

// ============================================================================
// Pose callback — FAST RRT (no rewiring)
// ============================================================================

void RRT::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    if (!scan_received_) return;

    car_x_ = pose_msg->pose.pose.position.x;
    car_y_ = pose_msg->pose.pose.position.y;
    double qx = pose_msg->pose.pose.orientation.x, qy = pose_msg->pose.pose.orientation.y;
    double qz = pose_msg->pose.pose.orientation.z, qw = pose_msg->pose.pose.orientation.w;
    car_yaw_ = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));

    if (use_waypoints_) choose_goal_from_waypoints();

    std::vector<RRT_Node> tree;
    tree.reserve(max_rrt_iters_ + 1);
    tree.push_back({0.0, 0.0, 0.0, -1, true});
    double gx = goal_x_, gy = goal_y_;

    // Goal marker
    visualization_msgs::msg::Marker gm;
    gm.header.frame_id = "base_link"; gm.header.stamp = this->now();
    gm.type = visualization_msgs::msg::Marker::SPHERE;
    gm.pose.position.x = gx; gm.pose.position.y = gy;
    gm.scale.x = 0.3; gm.scale.y = 0.3; gm.scale.z = 0.3;
    gm.color.b = 1.0; gm.color.a = 1.0;
    goal_viz_pub_->publish(gm);

    // ---- FAST RRT: no near(), no rewiring ----
    // Each iteration: sample → nearest → steer → collision check → add
    // Cost: O(n) for nearest only. No neighbor scanning.
    bool path_found = false;
    int goal_node_idx = -1;
    std::uniform_real_distribution<> bias_dist(0.0, 1.0);
    bool goal_blocked = is_occupied_wide(gx, gy, car_half_width_);

    for (int iter = 0; iter < max_rrt_iters_; ++iter) {
        // Sample with goal bias
        double bias = goal_blocked ? 0.08 : goal_bias_;
        std::vector<double> sampled = (bias_dist(gen) < bias)
            ? std::vector<double>{gx, gy} : sample();

        // Find nearest node in tree
        int nearest_idx = nearest(tree, sampled);

        // Steer toward sample
        RRT_Node new_node = steer(tree[nearest_idx], sampled);

        // Quick point check (fast reject)
        if (is_occupied_wide(new_node.x, new_node.y, car_half_width_)) continue;

        // Edge collision check
        if (check_collision(tree[nearest_idx], new_node)) continue;

        // Add to tree — simple parent, no rewiring
        new_node.parent = nearest_idx;
        new_node.cost = tree[nearest_idx].cost +
            std::sqrt(std::pow(new_node.x - tree[nearest_idx].x, 2) +
                      std::pow(new_node.y - tree[nearest_idx].y, 2));
        tree.push_back(new_node);

        if (is_goal(new_node, gx, gy)) {
            path_found = true;
            goal_node_idx = tree.size() - 1;
            break;
        }
    }

    // Partial path fallback
    if (!path_found && tree.size() > 5) {
        // Pass 1: good clearance, closest to goal
        double best = 1e9;
        for (size_t i = 1; i < tree.size(); ++i) {
            double wd = wall_distance(tree[i].x, tree[i].y);
            if (wd < car_half_width_ * 1.5 || tree[i].x < 0.1) continue;
            double d2g = std::pow(tree[i].x - gx, 2) + std::pow(tree[i].y - gy, 2);
            double score = d2g - std::min(wd, 1.0) * 0.3;
            if (score < best) { best = score; goal_node_idx = i; }
        }
        // Pass 2: relaxed — bare minimum clearance
        if (goal_node_idx <= 0) {
            best = 1e9;
            for (size_t i = 1; i < tree.size(); ++i) {
                double wd = wall_distance(tree[i].x, tree[i].y);
                if (wd < car_half_width_ || tree[i].x < 0.05) continue;
                double d2g = std::pow(tree[i].x - gx, 2) + std::pow(tree[i].y - gy, 2);
                if (d2g < best) { best = d2g; goal_node_idx = i; }
            }
        }
        if (goal_node_idx > 0) path_found = true;
    }

    // Visualize tree
    visualization_msgs::msg::MarkerArray tm;
    visualization_msgs::msg::Marker edges;
    edges.header.frame_id = "base_link"; edges.header.stamp = this->now();
    edges.type = visualization_msgs::msg::Marker::LINE_LIST;
    edges.scale.x = 0.01; edges.color.g = 1.0; edges.color.a = 0.4;
    for (size_t i = 1; i < tree.size(); ++i) {
        if (tree[i].parent < 0) continue;
        geometry_msgs::msg::Point p1, p2;
        p1.x = tree[tree[i].parent].x; p1.y = tree[tree[i].parent].y;
        p2.x = tree[i].x; p2.y = tree[i].y;
        edges.points.push_back(p1); edges.points.push_back(p2);
    }
    tm.markers.push_back(edges);
    tree_viz_pub_->publish(tm);

    // Drive
    if (!path_found) {
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.drive.speed = 0.0; msg.drive.steering_angle = 0.0;
        drive_pub_->publish(msg);
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "RRT: STOPPED — no path, tree=%zu", tree.size());
        return;
    }

    std::vector<RRT_Node> path = find_path(tree, tree[goal_node_idx]);

    // Visualize path
    visualization_msgs::msg::Marker pm;
    pm.header.frame_id = "base_link"; pm.header.stamp = this->now();
    pm.type = visualization_msgs::msg::Marker::LINE_STRIP;
    pm.scale.x = 0.05;
    pm.color.r = 1.0; pm.color.g = 0.2; pm.color.b = 0.2; pm.color.a = 1.0;
    for (auto &n : path) {
        geometry_msgs::msg::Point p; p.x = n.x; p.y = n.y;
        pm.points.push_back(p);
    }
    path_viz_pub_->publish(pm);

    // Pure pursuit
    double tx = path.back().x, ty = path.back().y;
    for (auto &n : path) {
        if (std::sqrt(n.x * n.x + n.y * n.y) >= lookahead_) { tx = n.x; ty = n.y; break; }
    }

    double L2 = tx * tx + ty * ty;
    double steering = std::atan((2.0 * ty / std::max(L2, 0.001)) * 0.3302);
    steering = std::clamp(steering, -0.4189, 0.4189);

    auto msg = ackermann_msgs::msg::AckermannDriveStamped();
    msg.drive.steering_angle = steering;
    msg.drive.speed = (std::abs(steering) > steer_threshold_) ? speed_corner_ : speed_straight_;
    drive_pub_->publish(msg);
}

// ============================================================================
// RRT methods
// ============================================================================

std::vector<double> RRT::sample() {
    for (int a = 0; a < 30; ++a) {
        double sx = x_dist(gen), sy = y_dist(gen);
        if (wall_distance(sx, sy) > car_half_width_) return {sx, sy};
    }
    return {x_dist(gen), y_dist(gen)};
}

int RRT::nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled) {
    int nid = 0; double min_d = 1e9;
    for (size_t i = 0; i < tree.size(); ++i) {
        double d = (tree[i].x - sampled[0]) * (tree[i].x - sampled[0]) +
                   (tree[i].y - sampled[1]) * (tree[i].y - sampled[1]);
        if (d < min_d) { min_d = d; nid = i; }
    }
    return nid;
}

RRT_Node RRT::steer(RRT_Node &nearest, std::vector<double> &sampled) {
    double dx = sampled[0] - nearest.x, dy = sampled[1] - nearest.y;
    double d = std::sqrt(dx * dx + dy * dy);
    if (d < 1e-6) return {nearest.x, nearest.y, 0.0, -1, false};
    double step = std::min(d, max_expansion_dist_);
    return {nearest.x + (dx / d) * step, nearest.y + (dy / d) * step, 0.0, -1, false};
}

bool RRT::check_collision(RRT_Node &n1, RRT_Node &n2) {
    double dx = n2.x - n1.x, dy = n2.y - n1.y;
    double d = std::sqrt(dx * dx + dy * dy);
    // Step at grid resolution (not half — faster)
    int steps = static_cast<int>(std::ceil(d / GRID_RESOLUTION));
    for (int i = 0; i <= steps; ++i) {
        double t = static_cast<double>(i) / std::max(steps, 1);
        if (is_occupied_wide(n1.x + t * dx, n1.y + t * dy, car_half_width_)) return true;
    }
    return false;
}

bool RRT::is_goal(RRT_Node &n, double gx, double gy) {
    return (n.x - gx) * (n.x - gx) + (n.y - gy) * (n.y - gy) < goal_threshold_ * goal_threshold_;
}

std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, RRT_Node &last) {
    std::vector<RRT_Node> path;
    int curr = -1;
    for (size_t i = 0; i < tree.size(); ++i)
        if (&tree[i] == &last) { curr = i; break; }
    while (curr != -1) { path.push_back(tree[curr]); curr = tree[curr].parent; }
    std::reverse(path.begin(), path.end());
    return path;
}

double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
    double d = std::sqrt((n1.x - n2.x) * (n1.x - n2.x) + (n1.y - n2.y) * (n1.y - n2.y));
    double wd = wall_distance((n1.x + n2.x) / 2.0, (n1.y + n2.y) / 2.0);
    return d + (wd < 1.0 ? wall_cost_weight_ * d * (1.0 - wd) : 0.0);
}