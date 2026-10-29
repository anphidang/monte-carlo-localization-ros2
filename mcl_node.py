#!/usr/bin/env python3
"""
MCL Node - Monte Carlo Localization ROS2 Node

This node integrates all MCL components:
- Subscribes to /robot_noisy (odometry) for motion updates
- Subscribes to /landmarks_observed (PointCloud2) for measurement updates
- Subscribes to /landmarks_gt (PointCloud2) for known landmark map
- Publishes /mcl/estimated_pose (PoseStamped)
- Publishes /mcl/particles (PoseArray) for RViz visualization
- Publishes /mcl/particle_cloud (PointCloud2) alternative visualization
"""

import rclpy
from rclpy.node import Node
import numpy as np
import struct
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ROS2 message types
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, PoseWithCovarianceStamped
from std_msgs.msg import Header

# Local imports
from .particle_filter import ParticleFilter, Particle


class MCLNode(Node):
    """
    Monte Carlo Localization Node.

    Combines odometry and landmark observations to estimate robot pose.
    """

    def __init__(self):
        super().__init__('mcl_node')

        # =====================================================================
        # Parameters
        # =====================================================================
        self.declare_parameter('num_particles', 1000)
        self.declare_parameter('map_x_min', -12.0)
        self.declare_parameter('map_x_max', 12.0)
        self.declare_parameter('map_y_min', -12.0)
        self.declare_parameter('map_y_max', 12.0)

        # Motion noise parameters
        self.declare_parameter('alpha1', 0.1)  # Rotation noise from rotation
        self.declare_parameter('alpha2', 0.1)  # Rotation noise from translation
        self.declare_parameter('alpha3', 0.1)  # Translation noise from translation
        self.declare_parameter('alpha4', 0.1)  # Translation noise from rotation

        # Sensor noise parameter
        self.declare_parameter('sensor_sigma', 0.5)

        # Resampling threshold (Neff / N)
        self.declare_parameter('resample_threshold', 0.5)

        # =====================================================================
        # Evaluation Parameters
        # =====================================================================
        self.declare_parameter('run_id', '')  # Identifier for this run
        self.declare_parameter('run_duration', 60.0)  # Duration in seconds (0 = infinite)
        self.declare_parameter('output_dir', '')  # Directory for CSV output
        self.declare_parameter('convergence_threshold', 0.5)  # Error threshold in meters
        self.declare_parameter('convergence_window', 2.0)  # Time window in seconds

        # Get parameters
        num_particles = self.get_parameter('num_particles').value
        map_bounds = {
            'x_min': self.get_parameter('map_x_min').value,
            'x_max': self.get_parameter('map_x_max').value,
            'y_min': self.get_parameter('map_y_min').value,
            'y_max': self.get_parameter('map_y_max').value,
        }

        self.get_logger().info(f'Initializing MCL with {num_particles} particles')
        self.get_logger().info(f'Map bounds: {map_bounds}')

        # =====================================================================
        # Initialize Particle Filter
        # =====================================================================
        self.pf = ParticleFilter(num_particles, map_bounds)

        # Set noise parameters
        self.pf.set_motion_noise(
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        )
        self.pf.set_sensor_sigma(self.get_parameter('sensor_sigma').value)

        self.resample_threshold = self.get_parameter('resample_threshold').value

        # =====================================================================
        # State variables
        # =====================================================================
        self.prev_odom: Optional[Odometry] = None
        self.known_landmarks: Dict[int, Tuple[float, float]] = {}
        self.landmarks_received = False

        # =====================================================================
        # Evaluation State
        # =====================================================================
        self.latest_gt: Optional[Odometry] = None
        self.errors: List[float] = []  # Position errors over time
        self.error_timestamps: List[float] = []  # Timestamps for errors
        self.start_time: Optional[float] = None
        self.convergence_time: Optional[float] = None
        self.converged = False
        self.consecutive_converged_samples = 0

        # Get experiment parameters
        self.run_id = self.get_parameter('run_id').value
        self.run_duration = self.get_parameter('run_duration').value
        self.output_dir = self.get_parameter('output_dir').value
        self.convergence_threshold = self.get_parameter('convergence_threshold').value
        self.convergence_window = self.get_parameter('convergence_window').value

        # Samples needed for convergence (at 10 Hz)
        self.convergence_samples_needed = int(self.convergence_window * 10)

        # =====================================================================
        # Subscribers
        # =====================================================================
        # Noisy odometry for motion update
        self.odom_sub = self.create_subscription(
            Odometry,
            '/robot_noisy',
            self.odom_callback,
            10
        )

        # Observed landmarks for measurement update
        self.landmarks_obs_sub = self.create_subscription(
            PointCloud2,
            '/landmarks_observed',
            self.landmarks_observed_callback,
            10
        )

        # Ground truth landmarks for map
        self.landmarks_gt_sub = self.create_subscription(
            PointCloud2,
            '/landmarks_gt',
            self.landmarks_gt_callback,
            10
        )

        # Ground truth robot pose for error calculation
        self.gt_sub = self.create_subscription(
            Odometry,
            '/robot_gt',
            self.gt_callback,
            10
        )

        # =====================================================================
        # Publishers
        # =====================================================================
        # Estimated pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/mcl/estimated_pose',
            10
        )

        # Pose with covariance (for more detailed visualization)
        self.pose_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/mcl/estimated_pose_cov',
            10
        )

        # Particle cloud as PoseArray (for RViz)
        self.particles_pub = self.create_publisher(
            PoseArray,
            '/mcl/particles',
            10
        )

        # Particle cloud as PointCloud2 (alternative visualization)
        self.particle_cloud_pub = self.create_publisher(
            PointCloud2,
            '/mcl/particle_cloud',
            10
        )

        # =====================================================================
        # Timer for periodic publishing (10 Hz)
        # =====================================================================
        self.timer = self.create_timer(0.1, self.publish_visualization)

        self.get_logger().info('MCL Node initialized successfully')

    # =========================================================================
    # Callbacks
    # =========================================================================
    def odom_callback(self, msg: Odometry) -> None:
        """
        Handle odometry messages for motion update.

        Computes delta motion from previous odometry and applies to particles.
        """
        if self.prev_odom is None:
            self.prev_odom = msg
            return

        # Compute delta motion in the odometry frame
        dx_odom = msg.pose.pose.position.x - self.prev_odom.pose.pose.position.x
        dy_odom = msg.pose.pose.position.y - self.prev_odom.pose.pose.position.y

        # Extract yaw from quaternion
        curr_yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)
        prev_yaw = self._quaternion_to_yaw(self.prev_odom.pose.pose.orientation)
        delta_theta = self._normalize_angle(curr_yaw - prev_yaw)

        # Transform the increment into the robot frame of the previous pose,
        # so it is independent of the odometry frame's orientation
        cos_p = np.cos(prev_yaw)
        sin_p = np.sin(prev_yaw)
        delta_x = cos_p * dx_odom + sin_p * dy_odom
        delta_y = -sin_p * dx_odom + cos_p * dy_odom

        # Only update if there's significant motion
        if abs(delta_x) > 1e-6 or abs(delta_y) > 1e-6 or abs(delta_theta) > 1e-6:
            self.pf.motion_update(delta_x, delta_y, delta_theta)

        self.prev_odom = msg

    def landmarks_gt_callback(self, msg: PointCloud2) -> None:
        """
        Handle ground truth landmarks message.

        Builds the known landmark map used for measurement update.
        Only processed once (or when landmarks change).
        """
        if self.landmarks_received:
            return

        landmarks = self._parse_pointcloud2(msg)
        for x, y, landmark_id in landmarks:
            self.known_landmarks[landmark_id] = (x, y)

        self.landmarks_received = True
        self.get_logger().info(f'Received {len(self.known_landmarks)} landmarks')

    def landmarks_observed_callback(self, msg: PointCloud2) -> None:
        """
        Handle observed landmarks message for measurement update.

        Landmarks are in robot frame - will be transformed using particle poses.
        """
        if not self.landmarks_received:
            return

        observed = self._parse_pointcloud2(msg)

        if observed:
            # Measurement update
            self.pf.measurement_update(observed, self.known_landmarks)

            # Adaptive resampling based on effective particle count
            neff = self.pf.get_effective_particle_count()
            if neff < self.resample_threshold * len(self.pf.particles):
                self.pf.low_variance_resample()

    def gt_callback(self, msg: Odometry) -> None:
        """
        Handle ground truth robot pose for error calculation.
        """
        self.latest_gt = msg

    # =========================================================================
    # Publishing
    # =========================================================================
    def publish_visualization(self) -> None:
        """Publish estimated pose and particle visualization."""
        now = self.get_clock().now()
        now_msg = now.to_msg()

        # Initialize start time on first call
        if self.start_time is None:
            self.start_time = now.nanoseconds / 1e9

        current_time = now.nanoseconds / 1e9
        elapsed = current_time - self.start_time

        # Estimate current pose
        x, y, theta = self.pf.estimate_pose()

        # =====================================================================
        # Error Calculation
        # =====================================================================
        if self.latest_gt is not None:
            gt_x = self.latest_gt.pose.pose.position.x
            gt_y = self.latest_gt.pose.pose.position.y
            err = np.sqrt((x - gt_x)**2 + (y - gt_y)**2)

            self.errors.append(err)
            self.error_timestamps.append(elapsed)

            # Check convergence
            if err < self.convergence_threshold:
                self.consecutive_converged_samples += 1
                if (not self.converged and
                    self.consecutive_converged_samples >= self.convergence_samples_needed):
                    self.converged = True
                    self.convergence_time = elapsed
                    self.get_logger().info(
                        f'Converged at t={elapsed:.2f}s (error < {self.convergence_threshold}m '
                        f'for {self.convergence_window}s)')
            else:
                self.consecutive_converged_samples = 0

        # Check if run duration exceeded
        if self.run_duration > 0 and elapsed >= self.run_duration:
            self._save_results_and_shutdown()

        # Publish estimated pose
        self._publish_pose(x, y, theta, now_msg)

        # Publish pose with covariance
        self._publish_pose_with_covariance(x, y, theta, now_msg)

        # Publish particles as PoseArray
        self._publish_particles_pose_array(now_msg)

        # Publish particles as PointCloud2
        self._publish_particles_pointcloud(now_msg)

    def _publish_pose(self, x: float, y: float, theta: float, stamp) -> None:
        """Publish estimated pose as PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0

        # Convert yaw to quaternion
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = np.sin(theta / 2.0)
        msg.pose.orientation.w = np.cos(theta / 2.0)

        self.pose_pub.publish(msg)

    def _publish_pose_with_covariance(self, x: float, y: float, theta: float, stamp) -> None:
        """Publish estimated pose with covariance."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = np.sin(theta / 2.0)
        msg.pose.pose.orientation.w = np.cos(theta / 2.0)

        # Compute covariance from particle distribution
        cov = self.pf.compute_covariance()

        # Fill 6x6 covariance matrix (x, y, z, roll, pitch, yaw)
        # We only have x, y, yaw so fill those
        msg.pose.covariance[0] = cov[0, 0]   # xx
        msg.pose.covariance[1] = cov[0, 1]   # xy
        msg.pose.covariance[5] = cov[0, 2]   # x-yaw
        msg.pose.covariance[6] = cov[1, 0]   # yx
        msg.pose.covariance[7] = cov[1, 1]   # yy
        msg.pose.covariance[11] = cov[1, 2]  # y-yaw
        msg.pose.covariance[30] = cov[2, 0]  # yaw-x
        msg.pose.covariance[31] = cov[2, 1]  # yaw-y
        msg.pose.covariance[35] = cov[2, 2]  # yaw-yaw

        self.pose_cov_pub.publish(msg)

    def _publish_particles_pose_array(self, stamp) -> None:
        """Publish particles as PoseArray for RViz visualization."""
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'

        for p in self.pf.get_particles():
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.position.z = 0.0
            pose.orientation.z = np.sin(p.theta / 2.0)
            pose.orientation.w = np.cos(p.theta / 2.0)
            msg.poses.append(pose)

        self.particles_pub.publish(msg)

    def _publish_particles_pointcloud(self, stamp) -> None:
        """Publish particles as PointCloud2 with weight as intensity."""
        particles = self.pf.get_particles()
        n = len(particles)

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.height = 1
        msg.width = n

        # Fields: x, y, z, weight
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.point_step = 16
        msg.row_step = msg.point_step * n
        msg.is_dense = True
        msg.data = bytearray(msg.row_step)

        for i, p in enumerate(particles):
            offset = i * msg.point_step
            struct.pack_into('ffff', msg.data, offset, p.x, p.y, 0.0, p.weight)

        self.particle_cloud_pub.publish(msg)

    # =========================================================================
    # Evaluation Methods
    # =========================================================================
    def compute_rmse(self) -> float:
        """Compute Root Mean Square Error over all collected errors."""
        if not self.errors:
            return float('nan')
        return float(np.sqrt(np.mean(np.square(self.errors))))

    def compute_rmse_after_convergence(self) -> float:
        """Compute RMSE only after convergence (steady-state error)."""
        if not self.converged or self.convergence_time is None:
            return float('nan')

        # Find errors after convergence
        post_convergence_errors = [
            err for err, t in zip(self.errors, self.error_timestamps)
            if t >= self.convergence_time
        ]

        if not post_convergence_errors:
            return float('nan')
        return float(np.sqrt(np.mean(np.square(post_convergence_errors))))

    def _save_results_and_shutdown(self) -> None:
        """Save experiment results to CSV and shutdown."""
        if not self.output_dir:
            self.get_logger().info('No output directory specified, skipping CSV save')
            self._log_summary()
            raise SystemExit(0)

        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate run_id if not specified
        run_id = self.run_id if self.run_id else datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed error log
        error_file = os.path.join(self.output_dir, f'errors_{run_id}.csv')
        with open(error_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'error'])
            for t, err in zip(self.error_timestamps, self.errors):
                writer.writerow([f'{t:.3f}', f'{err:.6f}'])
        self.get_logger().info(f'Saved error log to {error_file}')

        # Append to summary file
        summary_file = os.path.join(self.output_dir, 'summary.csv')
        write_header = not os.path.exists(summary_file)

        num_particles = self.get_parameter('num_particles').value
        alpha1 = self.get_parameter('alpha1').value
        alpha2 = self.get_parameter('alpha2').value
        alpha3 = self.get_parameter('alpha3').value
        alpha4 = self.get_parameter('alpha4').value
        sensor_sigma = self.get_parameter('sensor_sigma').value

        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'run_id', 'num_particles', 'alpha1', 'alpha2', 'alpha3', 'alpha4',
                    'sensor_sigma', 'rmse', 'rmse_after_convergence',
                    'convergence_time', 'converged', 'run_duration'
                ])
            writer.writerow([
                run_id,
                num_particles,
                alpha1, alpha2, alpha3, alpha4,
                sensor_sigma,
                f'{self.compute_rmse():.6f}',
                f'{self.compute_rmse_after_convergence():.6f}',
                f'{self.convergence_time:.3f}' if self.convergence_time else 'nan',
                self.converged,
                f'{self.error_timestamps[-1]:.3f}' if self.error_timestamps else '0'
            ])
        self.get_logger().info(f'Appended summary to {summary_file}')

        self._log_summary()
        raise SystemExit(0)

    def _log_summary(self) -> None:
        """Log experiment summary to console."""
        self.get_logger().info('=' * 60)
        self.get_logger().info('EXPERIMENT SUMMARY')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Run ID: {self.run_id}')
        self.get_logger().info(f'Particles: {self.get_parameter("num_particles").value}')
        self.get_logger().info(f'Sensor Sigma: {self.get_parameter("sensor_sigma").value}')
        self.get_logger().info(f'Alpha1-4: {self.get_parameter("alpha1").value}, '
                               f'{self.get_parameter("alpha2").value}, '
                               f'{self.get_parameter("alpha3").value}, '
                               f'{self.get_parameter("alpha4").value}')
        self.get_logger().info(f'RMSE: {self.compute_rmse():.4f} m')
        self.get_logger().info(f'RMSE (after convergence): {self.compute_rmse_after_convergence():.4f} m')
        self.get_logger().info(f'Converged: {self.converged}')
        if self.convergence_time:
            self.get_logger().info(f'Convergence Time: {self.convergence_time:.2f} s')
        self.get_logger().info('=' * 60)

    # =========================================================================
    # Helper methods
    # =========================================================================
    def _parse_pointcloud2(self, msg: PointCloud2) -> List[Tuple[float, float, int]]:
        """
        Parse PointCloud2 message to list of (x, y, id) tuples.

        Assumes the format from fake_robot: x, y, z (float32), id (int32)
        """
        points = []
        point_step = msg.point_step

        for i in range(msg.width):
            offset = i * point_step
            x, y, z = struct.unpack_from('fff', msg.data, offset)
            landmark_id = struct.unpack_from('i', msg.data, offset + 12)[0]
            points.append((x, y, landmark_id))

        return points

    def _quaternion_to_yaw(self, q) -> float:
        """Extract yaw angle from quaternion."""
        # For 2D (z-axis rotation only): yaw = 2 * atan2(qz, qw)
        return 2.0 * np.arctan2(q.z, q.w)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = MCLNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
