#!/usr/bin/env python3
"""
Particle Filter Implementation for Monte Carlo Localization (MCL)

Components:
- Uniform particle initialization (global localization)
- Odometry-based motion update with motion-dependent Gaussian noise
- Landmark-based measurement update (log-likelihood, numerically stable)
- Low-variance (systematic) resampling
- Weighted mean pose estimation (circular mean for orientation)
- Helpers for visualization (particle arrays, covariance, pose history)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Particle:
    """
    Represents a single particle in the particle filter.

    Attributes:
        x: x-position in map frame (meters)
        y: y-position in map frame (meters)
        theta: orientation in map frame (radians, range [-π, π])
        weight: particle weight (probability)
    """
    x: float
    y: float
    theta: float
    weight: float


class ParticleFilter:
    """
    Particle Filter for Monte Carlo Localization.
    """

    def __init__(self, num_particles: int, map_bounds: Dict[str, float]):
        """
        Initialize the particle filter.

        Args:
            num_particles: Number of particles to use
            map_bounds: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'
        """
        self.num_particles = num_particles
        self.map_bounds = map_bounds
        self.particles: List[Particle] = []

        # Sensor model parameter
        self.sensor_sigma = 0.5  # Standard deviation for likelihood calculation

        # Motion model noise parameters (can be tuned)
        self.alpha1 = 0.1  # Rotation noise from rotation
        self.alpha2 = 0.1  # Rotation noise from translation
        self.alpha3 = 0.1  # Translation noise from translation
        self.alpha4 = 0.1  # Translation noise from rotation

        # Pose history for trajectory visualization
        self.pose_history: List[Tuple[float, float, float]] = []
        self.max_history_length = 100

        # Initialize particles
        self.initialize_particles()

    # =========================================================================
    # Particle Initialization
    # =========================================================================
    def initialize_particles(self) -> None:
        """
        Uniformly distribute particles across the map.

        - Particles are distributed uniformly in x, y within map_bounds
        - Orientation theta is uniformly distributed in [-π, π]
        - All particles receive equal weights (1/N)
        """
        self.particles = []
        initial_weight = 1.0 / self.num_particles

        for _ in range(self.num_particles):
            # Uniform distribution for position
            x = np.random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max'])
            y = np.random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max'])

            # Uniform distribution for orientation
            theta = np.random.uniform(-np.pi, np.pi)

            self.particles.append(Particle(x, y, theta, initial_weight))

        # Clear pose history on re-initialization
        self.pose_history = []

    # =========================================================================
    # Motion Update
    # =========================================================================
    def motion_update(self, delta_x: float, delta_y: float, delta_theta: float) -> None:
        """
        Apply odometry-based motion model to each particle.

        The odometry increment is given in the ROBOT frame (relative to the
        previous robot pose). Each particle applies this increment relative
        to its own hypothesized heading, so particles with different
        orientations move in different map directions.

        Gaussian noise is added whose standard deviation scales with the
        magnitude of translation and rotation (alpha1..alpha4).

        Args:
            delta_x: Forward motion in robot frame since last update
            delta_y: Lateral motion in robot frame since last update
            delta_theta: Change in orientation since last update
        """
        # Compute motion magnitude for noise scaling
        translation = np.sqrt(delta_x**2 + delta_y**2)
        rotation = abs(delta_theta)

        trans_std = self.alpha3 * translation + self.alpha4 * rotation + 1e-6
        rot_std = self.alpha1 * rotation + self.alpha2 * translation + 1e-6

        for particle in self.particles:
            # Noisy increment in the robot frame
            dx = delta_x + np.random.normal(0, trans_std)
            dy = delta_y + np.random.normal(0, trans_std)
            dtheta = delta_theta + np.random.normal(0, rot_std)

            # Rotate robot-frame increment into the map frame
            # using the particle's own heading
            cos_t = np.cos(particle.theta)
            sin_t = np.sin(particle.theta)
            particle.x += cos_t * dx - sin_t * dy
            particle.y += sin_t * dx + cos_t * dy
            particle.theta += dtheta

            # Normalize orientation to [-π, π]
            particle.theta = self._normalize_angle(particle.theta)

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to range [-π, π].

        Uses atan2(sin, cos) which is numerically stable and
        handles all edge cases correctly.
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def set_motion_noise(self, alpha1: float, alpha2: float,
                         alpha3: float, alpha4: float) -> None:
        """
        Set motion model noise parameters.

        Args:
            alpha1: Rotation noise from rotation
            alpha2: Rotation noise from translation
            alpha3: Translation noise from translation
            alpha4: Translation noise from rotation
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

    # =========================================================================
    # Measurement Update
    # =========================================================================
    def measurement_update(self,
                           observed_landmarks: List[Tuple[float, float, int]],
                           known_landmarks: Dict[int, Tuple[float, float]]) -> None:
        """
        Update particle weights based on landmark observations.

        Landmark-based sensor model with known data association:
        - Transform each observed landmark into the map frame using the particle pose
        - Compare with the known landmark position
        - Weight is the product of Gaussian likelihoods

        Computation is done in log space to avoid numerical underflow when
        many landmarks are observed or sensor_sigma is small.

        Args:
            observed_landmarks: List of (x_robot, y_robot, landmark_id) in robot frame
            known_landmarks: Dictionary mapping landmark_id -> (x_map, y_map)
        """
        if not observed_landmarks:
            return

        log_weights = np.zeros(len(self.particles))

        for idx, particle in enumerate(self.particles):
            log_likelihood = 0.0

            cos_theta = np.cos(particle.theta)
            sin_theta = np.sin(particle.theta)

            for obs_x, obs_y, obs_id in observed_landmarks:
                if obs_id not in known_landmarks:
                    continue

                # Transform observation from robot frame to map frame
                # using the particle's hypothesized pose
                map_x = particle.x + cos_theta * obs_x - sin_theta * obs_y
                map_y = particle.y + sin_theta * obs_x + cos_theta * obs_y

                # Get expected landmark position from map
                expected_x, expected_y = known_landmarks[obs_id]

                # Squared Euclidean distance
                dist_sq = (map_x - expected_x)**2 + (map_y - expected_y)**2

                # Gaussian log-likelihood (constant terms omitted)
                log_likelihood += -0.5 * dist_sq / self.sensor_sigma**2

            log_weights[idx] = np.log(max(particle.weight, 1e-300)) + log_likelihood

        # Subtract max before exponentiating (log-sum-exp trick)
        log_weights -= np.max(log_weights)
        for particle, lw in zip(self.particles, log_weights):
            particle.weight = float(np.exp(lw))

        # Normalize weights
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize particle weights to sum to 1."""
        total_weight = sum(p.weight for p in self.particles)

        if total_weight > 1e-10:  # Avoid division by zero
            for p in self.particles:
                p.weight /= total_weight
        else:
            # If all weights are zero, reset to uniform
            uniform_weight = 1.0 / self.num_particles
            for p in self.particles:
                p.weight = uniform_weight

    def set_sensor_sigma(self, sigma: float) -> None:
        """Set sensor model standard deviation."""
        self.sensor_sigma = sigma

    # =========================================================================
    # Resampling
    # =========================================================================
    def low_variance_resample(self) -> None:
        """
        Low-variance (systematic) resampling.

        Advantages over multinomial resampling:
        - Linear time O(N)
        - Lower variance in particle selection
        - A particle set with uniform weights is preserved unchanged
        """
        n = len(self.particles)
        new_particles = []

        # Random start point
        r = np.random.uniform(0, 1.0 / n)

        # Cumulative weight
        c = self.particles[0].weight
        i = 0

        for m in range(n):
            # Target cumulative weight
            U = r + m / n

            # Find particle whose cumulative weight encompasses U
            while U > c and i < n - 1:
                i += 1
                c += self.particles[i].weight

            # Create copy of selected particle with reset weight
            new_particles.append(Particle(
                x=self.particles[i].x,
                y=self.particles[i].y,
                theta=self.particles[i].theta,
                weight=1.0 / n  # Equal weights after resampling
            ))

        self.particles = new_particles

    # =========================================================================
    # Pose Estimation
    # =========================================================================
    def estimate_pose(self, record_history: bool = True) -> Tuple[float, float, float]:
        """
        Compute weighted mean of all particles.

        For orientation, we use circular mean (atan2 of weighted sin/cos)
        to properly handle angle wraparound.

        Args:
            record_history: If True, append the estimate to the pose history

        Returns:
            Tuple of (x, y, theta) representing estimated pose
        """
        # Weighted mean for position
        x_mean = sum(p.x * p.weight for p in self.particles)
        y_mean = sum(p.y * p.weight for p in self.particles)

        # Circular mean for orientation
        # This correctly handles the wraparound at ±π
        sin_mean = sum(np.sin(p.theta) * p.weight for p in self.particles)
        cos_mean = sum(np.cos(p.theta) * p.weight for p in self.particles)
        theta_mean = np.arctan2(sin_mean, cos_mean)

        # Store in history for trajectory visualization
        if record_history:
            self._add_to_history(x_mean, y_mean, theta_mean)

        return x_mean, y_mean, theta_mean

    # =========================================================================
    # Visualization Helpers
    # =========================================================================
    def _add_to_history(self, x: float, y: float, theta: float) -> None:
        """Add pose to history for trajectory visualization."""
        self.pose_history.append((x, y, theta))

        # Limit history length
        if len(self.pose_history) > self.max_history_length:
            self.pose_history.pop(0)

    def get_particles(self) -> List[Particle]:
        """Return current particle set for visualization."""
        return self.particles

    def get_particles_as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return particles as numpy arrays for efficient visualization.

        Returns:
            Tuple of (x_array, y_array, theta_array, weight_array)
        """
        n = len(self.particles)
        x = np.zeros(n)
        y = np.zeros(n)
        theta = np.zeros(n)
        weights = np.zeros(n)

        for i, p in enumerate(self.particles):
            x[i] = p.x
            y[i] = p.y
            theta[i] = p.theta
            weights[i] = p.weight

        return x, y, theta, weights

    def get_pose_history(self) -> List[Tuple[float, float, float]]:
        """Return pose history for trajectory visualization."""
        return self.pose_history

    def compute_covariance(self) -> np.ndarray:
        """
        Compute weighted covariance matrix of particle distribution.

        Useful for visualizing uncertainty ellipse in RViz.

        Returns:
            3x3 covariance matrix for [x, y, theta]
        """
        x_mean, y_mean, theta_mean = self.estimate_pose(record_history=False)

        cov = np.zeros((3, 3))

        for p in self.particles:
            dx = p.x - x_mean
            dy = p.y - y_mean
            # Handle angle difference properly
            dtheta = self._normalize_angle(p.theta - theta_mean)

            diff = np.array([dx, dy, dtheta])
            cov += p.weight * np.outer(diff, diff)

        return cov

    def get_effective_particle_count(self) -> float:
        """
        Compute effective number of particles (Neff).

        Low Neff indicates particle degeneracy and need for resampling.
        Neff = 1 / sum(weight^2)

        Returns:
            Effective particle count (1 to N)
        """
        weight_squared_sum = sum(p.weight**2 for p in self.particles)

        if weight_squared_sum > 1e-10:
            return 1.0 / weight_squared_sum
        else:
            return 0.0
