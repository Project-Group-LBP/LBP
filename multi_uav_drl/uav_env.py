import numpy as np
import gym
from gym import spaces

#  1. Move Hyperparameters Outside the Class
NUM_UAVS = 6
AREA_SIZE = 15  # Slightly larger area for better exploration
COMM_RANGE = 6  # Slightly larger comm range to encourage collaboration
MAX_STEPS = 500
OBSTACLE_PROBABILITY = 0.20  # Reduce obstacles slightly for better learning
COMMUNICATION_PENALTY_WEIGHT = 0.05  #  Increased to reinforce cooperation
OBSTACLE_PENALTY_WEIGHT = 0.05  #  Increasing for repeated mistakes
REWARD_DISCOUNT_FACTOR = 0.99  #  To reduce penalties for occasional mistakes
COVERAGE_REWARD_WEIGHT = 1.5  #  Give higher weight to spreading out

class MultiUAVEnv(gym.Env):
    def __init__(self):
        super(MultiUAVEnv, self).__init__()
        self.num_uavs = NUM_UAVS
        self.area_size = AREA_SIZE
        self.comm_range = COMM_RANGE
        self.max_steps = MAX_STEPS
        self.current_step = 0

        self.action_space = spaces.Box(low=np.zeros((self.num_uavs, 2), dtype=np.float32), 
                                       high=np.array([[2*np.pi, 1]] * self.num_uavs, dtype=np.float32), 
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.area_size, shape=(self.num_uavs, 2), dtype=np.float32)

        #  Track penalties over time for adaptive scaling
        # self.communication_violations = np.zeros(self.num_uavs)
        # self.obstacle_collisions = np.zeros(self.num_uavs)

        self.reset()

    def reset(self):
        self.uav_positions = np.random.uniform(0, self.area_size, (self.num_uavs, 2))
        self.map_obstacles = np.random.choice([0, 1], size=(self.area_size, self.area_size), 
                                              p=[1 - OBSTACLE_PROBABILITY, OBSTACLE_PROBABILITY])
        self.coverage = np.zeros((self.area_size, self.area_size))
        self.current_step = 0
        # self.communication_violations[:] = 0  # Reset penalties
        # self.obstacle_collisions[:] = 0
        return self.uav_positions

    def step(self, actions):
        prev_positions = np.copy(self.uav_positions)  # Store previous positions

        for i, action in enumerate(actions):
            angle, distance = action
            new_position = prev_positions[i] + np.array([distance * np.cos(angle), distance * np.sin(angle)])
            new_position = np.clip(new_position, 0, self.area_size - 1)

            x, y = int(new_position[0]), int(new_position[1])

            #  Allow movement but apply penalties for obstacles
            if self.map_obstacles[x, y] == 0:
                self.uav_positions[i] = new_position  # Move only if no obstacle
                self.coverage[x, y] += 1
            # else:
            #     self.obstacle_collisions[i] += 1  #  Track repeated obstacle hits

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self.uav_positions, done

    def _calculate_reward(self, prev_positions):
        """
        Computes the reward based on:
        - Coverage maximization
        - Fairness (Jain's Index)
        - Adaptive penalties for communication & obstacles
        - Movement efficiency reward (not sure)
        """
        coverage_score = self._calculate_coverage_reward()
        fairness = self._calculate_fairness()
        communication_penalty = self._calculate_adaptive_communication_penalty()
        obstacle_penalty = self._calculate_adaptive_obstacle_penalty()
        movement_efficiency = self._calculate_movement_efficiency(prev_positions)

        reward = coverage_score + fairness + movement_efficiency - communication_penalty - obstacle_penalty

        # Return individual components for logging/plotting outside
        return reward, (coverage_score, fairness, communication_penalty, obstacle_penalty, movement_efficiency)

    def _calculate_coverage_reward(self):
        """Encourages UAVs to spread out and maximize coverage"""
        return COVERAGE_REWARD_WEIGHT * (np.sum(self.coverage) / (self.area_size ** 2))

    def _calculate_fairness(self):
        """Computes Jain's fairness index for balanced coverage"""
        square_of_sum = np.square(np.sum(self.coverage))
        sum_of_squares = np.sum(np.square(self.coverage))
        return (square_of_sum / sum_of_squares) / float(len(self.coverage)) if sum_of_squares != 0 else 0

    def _calculate_adaptive_communication_penalty(self):
        """Penalizes UAVs more heavily if they keep breaking formation"""
        penalty = 0
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(self.uav_positions[i] - self.uav_positions[j])
                if distance > self.comm_range:
                    # self.communication_violations[i] += 1  #  Track repeated violations
                    # penalty += self.communication_violations[i]  #  Increase penalty over time
                    penalty += 1

        return COMMUNICATION_PENALTY_WEIGHT * penalty

    def _calculate_adaptive_obstacle_penalty(self):
        """Penalizes UAVs for colliding with obstacles or another UAV"""
        # return OBSTACLE_PENALTY_WEIGHT * np.sum(self.obstacle_collisions)  #  Increasing penalty over time
        penalty = 0
        for i in range(self.num_uavs):
            x, y = int(self.uav_positions[i][0]), int(self.uav_positions[i][1])
            if self.map_obstacles[x, y] == 1:
                penalty += 1  # Penalize UAV for hitting an obstacle
            for j in range(i+1, self.num_uavs):
                if((x, y) == int(self.uav_positions[j][0]), int(self.uav_positions[j][1])):
                    penalty += 1  # Penalize UAV for hitting another UAV

        return OBSTACLE_PENALTY_WEIGHT * penalty

    def _calculate_movement_efficiency(self, prev_positions):
        """Encourages UAVs to move efficiently rather than randomly"""
        total_distance = np.sum(np.linalg.norm(self.uav_positions - prev_positions, axis=1))
        return -0.01 * total_distance  #  Penalizes unnecessary movement
