# TODO: add obstacles and related collision computations

# TODO : expand observation space
# TODO : fix reward function

import numpy as np
import gymnasium
from gymnasium import spaces

#  Defining all the required Hyperparameters
NUM_UAVS = 6
AREA_SIZE = 10  # Slightly larger area for better exploration
COMM_RANGE = 5  # Beyond this range, UAV will break the network connection
COLLISION_RADIUS = 1  # Within this range, UAV-UAV collisions will take place
MAX_STEPS = 500  # Max number of steps allowed per episode
OBSTACLE_PROBABILITY = 0  # Prob with which the obstacles will be distributed in the environment
COMMUNICATION_PENALTY_WEIGHT = 1  #  Increased to reinforce cooperation
OBSTACLE_PENALTY_WEIGHT = 1  #  Increasing for repeated mistakes
ENERGY_PENALTY_WEIGHT = 1  #  For energy constraints
REWARD_DISCOUNT_FACTOR = 0.99  #  To reduce penalties for occasional mistakes
COVERAGE_REWARD_WEIGHT = 1.5  #  Give higher weight to spreading out
LOG_FREQUENCY = 10  # Logs will be printed after every LOG_FREQUENCY number of episodes


class MultiUAVEnv(gymnasium.Env):
    def __init__(self):
        super(MultiUAVEnv, self).__init__()
        self.num_uavs = NUM_UAVS
        self.area_size = AREA_SIZE
        self.comm_range = COMM_RANGE
        self.max_steps = MAX_STEPS
        self.log_freq = LOG_FREQUENCY
        self.current_step = 0

        self.action_space = spaces.Box(low=np.zeros((self.num_uavs, 2), dtype=np.float32), high=np.array([[2 * np.pi, 1]] * self.num_uavs, dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.area_size, shape=(self.num_uavs, 2), dtype=np.float32)

        #  Track penalties over time for adaptive scaling
        # self.communication_violations = np.zeros(self.num_uavs)
        self.obstacle_collisions = np.zeros(self.num_uavs)

        self.reset()

    def reset(self):
        self.uav_positions = np.random.uniform(0, self.area_size, (self.num_uavs, 2))
        self.map_obstacles = np.random.choice([0, 1], size=(self.area_size, self.area_size), p=[1 - OBSTACLE_PROBABILITY, OBSTACLE_PROBABILITY])
        self.coverage = np.zeros((self.area_size, self.area_size))
        self.current_step = 0

        return self.uav_positions

    def step(self, actions):
        prev_positions = np.copy(self.uav_positions)  # Store previous positions of each UAV
        prev_coverage = np.copy(self.coverage)  # Store previously covered cells

        for i, action in enumerate(actions):
            angle, distance = action
            intended_position = self.uav_positions[i] + np.array([distance * np.cos(angle), distance * np.sin(angle)])
            intended_position = np.clip(intended_position, 0, self.area_size - 1)

            x, y = int(intended_position[0]), int(intended_position[1])

            #  Allow movement but apply penalties for obstacles
            if self.map_obstacles[x, y] == 1:
                self.obstacle_collisions[i] += 1  # Collision happened with obstacle
            else:
                self.uav_positions[i] = intended_position  # Move only if no obstacle
                if self.coverage[x, y] == 0:
                    self.coverage[x, y] += 10  # Greater reword for exploring a new cell
                else:
                    self.coverage[x, y] += 1

        reward_per_uav, (cov, fair, energy_eff, comm_per_uav) = self._calculate_reward(prev_positions, prev_coverage)

        self.current_step += 1

        covered = not np.any(self.coverage == 0)
        done = covered or (self.current_step >= self.max_steps)

        return self.uav_positions, done, reward_per_uav, (cov, fair, energy_eff, comm_per_uav)

    def _calculate_reward(self, prev_positions, prev_coverage):
        """
        Reward = Δh_t - penalty
        Δh_t = (fairness * coverage_gain) / energy_spent
        penalty: obstacle collision or disconnected UAVs
        """
        coverage_score = self._calculate_coverage_reward(prev_coverage)
        fairness = self._calculate_fairness()
        energy_efficiency = self._calculate_energy_efficiency(prev_positions)

        comm_penalty_per_uav = self._calculate_communication_penalty()
        # obstacle_penalty_per_uav = self._calculate_obstacle_penalty()
        penalty_per_uav = comm_penalty_per_uav  # + obstacle_penalty_per_uav

        efficiency = (coverage_score * fairness) / (energy_efficiency if energy_efficiency != 0 else 1)
        reward_per_uav = [efficiency - penalty for penalty in penalty_per_uav]

        # Return individual components for logging/plotting outside
        return reward_per_uav, (coverage_score, fairness, energy_efficiency, comm_penalty_per_uav)

    def _calculate_coverage_reward(self, prev_coverage):
        """Encourages UAVs to spread out and maximize coverage"""
        # return COVERAGE_REWARD_WEIGHT * (np.sum(self.coverage) / (self.area_size ** 2))
        coverage_gain = np.sum(self.coverage - prev_coverage)
        return COVERAGE_REWARD_WEIGHT * coverage_gain

    def _calculate_fairness(self):
        """Computes Jain's fairness index for balanced coverage"""
        square_of_sum = np.square(np.sum(self.coverage))
        sum_of_squares = np.sum(np.square(self.coverage))
        return (square_of_sum / sum_of_squares) / (self.area_size**2) if sum_of_squares != 0 else 0

    def _calculate_communication_penalty(self):
        """Penalizes UAVs that lose local communication connectivity (nearest neighbor logic)"""
        penalty = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            is_connected = False
            for j in range(self.num_uavs):
                if j != i:
                    curr_pair_distance = np.linalg.norm(self.uav_positions[i] - self.uav_positions[j])
                    if curr_pair_distance <= self.comm_range:
                        is_connected = True
                        break

            if not is_connected:
                penalty[i] += 1

        return penalty

    # def _calculate_obstacle_penalty(self):
    #     """Penalizes UAVs for colliding with obstacles or another UAV"""
    #     obstacle_collision = np.sum(self.obstacle_collisions)
    #     collision_pairs = set()
    #     for i in range(self.num_uavs):
    #         for j in range(i + 1, self.num_uavs):
    #             if np.linalg.norm(self.uav_positions[i] - self.uav_positions[j]) < COLLISION_RADIUS:
    #                 collision_pairs.add((min(i, j), max(i, j)))

    #     uav_collision = len(collision_pairs)
    #     return OBSTACLE_PENALTY_WEIGHT * (obstacle_collision + uav_collision)

    def _calculate_energy_efficiency(self, prev_positions):
        """Encourages UAVs to move efficiently rather than randomly"""
        total_distance = np.sum(np.linalg.norm(self.uav_positions - prev_positions, axis=1))
        return ENERGY_PENALTY_WEIGHT * total_distance  #  Penalizes unnecessary movement
