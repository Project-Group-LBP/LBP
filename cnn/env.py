from data_points import test_data
import copy
import numpy as np
from gymnasium import spaces
from PIL import Image
import os

# Environment parameters
MAP_WIDTH = MAP_HEIGHT = 16
GRID_WIDTH = GRID_HEIGHT = 80
WALL_WIDTH = 4
WALL_VALUE = -1
CHANNEL = 3
NUM_UAVS = 6
MAX_ENERGY = 500
NUM_ACTION = 2
HOVER_ENERGY = 0.5
COMM_RANGE = 6.0
COV_RANGE = 3.0
MAX_DISTANCE = 3.0
WALL_PENALTY = -10.0
COMM_PENALTY = -1.0
EPSILON = 1e-4
ENERGY_FACTOR = 1.0 / 3.0
ENTROPY_FACTOR = 0.1


class Env:
    def __init__(self, image_init=False, log_dir="."):
        self.map_width = MAP_WIDTH
        self.map_height = MAP_HEIGHT
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.channels = CHANNEL
        self.img_path = log_dir
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        # UAV configuration
        self.num_uavs = NUM_UAVS
        self.init_positions = np.random.rand(self.num_uavs, 2) * self.map_width
        self.observation_space = [spaces.Box(low=-1, high=1, shape=(self.width, self.height, self.channels)) for _ in range(self.num_uavs)]
        self.action_space = [spaces.Box(low=-1, high=1, shape=(NUM_ACTION,)) for _ in range(self.num_uavs)]
        self.step_count = 0

        # Movement and collection parameters
        self.max_energy = MAX_ENERGY
        self.comm_range = COMM_RANGE
        self.cov_range = COV_RANGE
        self.max_dist = MAX_DISTANCE
        self.entropy_factor = ENTROPY_FACTOR  # Entropy factor for exploration
        self.energy_factor = ENERGY_FACTOR  # Energy needed per unit distance moved
        self.epsilon = EPSILON  # A small value for reference
        self.hover_energy = HOVER_ENERGY  # Energy needed for hovering
        self.p_wall = WALL_PENALTY
        self.p_comm = COMM_PENALTY
        self.dn = [False] * self.num_uavs  # UAVs with depleted energy
        self.energy = np.ones(self.num_uavs).astype(np.float64) * self.max_energy
        self.penalty = np.zeros(self.num_uavs)

        # Initialize data points from test_data module
        if image_init:
            from decoded_points import decoded_data  # type: ignore

            self.datas = np.reshape(decoded_data, (-1, 2)).astype(np.float16) * self.map_width
        else:
            self.datas = np.reshape(test_data, (-1, 2)).astype(np.float16) * self.map_width
        self.total_points = len(self.datas)
        self.coverage_map = np.zeros(self.total_points, dtype=bool)
        self.visit_count = np.zeros(self.total_points, dtype=np.int16)
        self.uav_pos = copy.deepcopy(self.init_positions)

        self._init_data_map = np.zeros((self.width, self.height)).astype(np.float16)
        self._init_position_map = np.zeros((self.num_uavs, self.width, self.height)).astype(np.float16)

        # Draw walls and data points on data map
        self._draw_wall(self._init_data_map)
        for pos in self.datas:
            self._draw_data_point(pos[0], pos[1], self._init_data_map)

        # Draw initial UAV positions
        for i_n in range(self.num_uavs):
            self._draw_UAV(self.init_positions[i_n][0], self.init_positions[i_n][1], 1.0, self._init_position_map[i_n])

    def _transform_coords(self, x, y):
        """Transform logical coordinates to visual coordinates"""
        return 4 * int(x) + WALL_WIDTH * 2, 4 * int(y) + WALL_WIDTH * 2

    def _draw_square(self, x, y, width, height, value, grid, add=False):
        for i in range(x, x + width):
            for j in range(y, y + height):
                if 0 <= i < self.width and 0 <= j < self.height:
                    if add:
                        grid[i][j] += value
                    else:
                        grid[i][j] = value

    def _draw_wall(self, grid):
        for j in range(self.height):
            for i in range(WALL_WIDTH):
                grid[i][j] = WALL_VALUE
            for i in range(self.height - WALL_WIDTH, self.height):
                grid[i][j] = WALL_VALUE
        for i in range(self.width):
            for j in range(WALL_WIDTH):
                grid[i][j] = WALL_VALUE
            for j in range(self.height - WALL_WIDTH, self.height):
                grid[i][j] = WALL_VALUE

    def _draw_data_point(self, x, y, grid, value=1.0):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 2, 2, value, grid, add=True)

    def _draw_UAV(self, x, y, value, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 4, 4, value, grid)

    def _clear_uav(self, x, y, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 4, 4, 0, grid)

    def save_state_image(self, name=None, include_uavs=True):
        grid = self._init_data_map.copy()
        max_value = np.max(grid)
        if max_value > 0:  # Normalize grid to [0, 1] range
            grid = grid / max_value
        rgb_img = np.stack([grid, grid, grid], axis=2)

        for i, pos in enumerate(self.datas):
            if self.coverage_map[i]:
                x, y = self._transform_coords(pos[0], pos[1])
                for dx in range(2):
                    for dy in range(2):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            rgb_img[x + dx, y + dy] = [0.7, 1.0, 0.7]
        if include_uavs:
            colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
            for i, pos in enumerate(self.uav_pos):
                x, y = self._transform_coords(pos[0], pos[1])
                color = colors[i % len(colors)]
                for dx in range(4):
                    for dy in range(4):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            rgb_img[x + dx, y + dy] = color
        img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img, "RGB")
        if name is None:
            name = "initial_state"
        img.save(os.path.join(self.img_path, f"{name}.png"), "png")

    def save_heat_map_image(self, name):
        cov_data = self.state[0][:, :, 2].copy()
        # Create RGB representation - initialize with black
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float64)
        
        # Make data points with zero coverage white
        data_points_mask = self._init_data_map > 0  # Mask for all data point locations
        zero_coverage_mask = (cov_data <= 0.01) & data_points_mask
        rgb_img[zero_coverage_mask] = 1.0  # Set to white
        
        # Blue to cyan to green to yellow to red colormap for covered points
        coverage_mask = cov_data > 0.01
        # Scale coverage from light blue to dark blue
        blue_intensity = 1.0 - (0.7 * cov_data[coverage_mask])  # Higher coverage = darker blue
        
        rgb_img[coverage_mask, 0] = blue_intensity  # Red channel
        rgb_img[coverage_mask, 1] = blue_intensity  # Green channel
        rgb_img[coverage_mask, 2] = 1.0  # Blue channel always at maximum
        
        # Convert to uint8 for PIL
        img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img, "RGB")
        img.save(os.path.join(self.img_path, f"{name}.png"), "png") 
        
    
    def __init_state(self):
        """Initialize state"""
        state = []
        for i in range(self.num_uavs):
            image = np.zeros((self.width, self.height, self.channels)).astype(np.float16)
            image[:, :, 0] = copy.copy(self._init_data_map)
            image[:, :, 1] = copy.copy(self._init_position_map[i])
            # image[:, :, 2] is already initialized to zeros
            state.append(image)
        self.state = state

    def __update_state(self, clear_uav_pos):
        """Update state with UAV positions and coverage info"""
        for n in range(self.num_uavs):
            # Update UAV positions (channel 1)
            self._clear_uav(clear_uav_pos[n][0], clear_uav_pos[n][1], self.state[n][:, :, 1])
            self._draw_UAV(self.uav_pos[n][0], self.uav_pos[n][1], self.energy[n] / self.max_energy, self.state[n][:, :, 1])

            # Update coverage information (channel 2)
            self.state[n][:, :, 2].fill(0.0)
            for i, pos in enumerate(self.datas):
                if self.visit_count[i] > 0:
                    self._draw_data_point(pos[0], pos[1], self.state[n][:, :, 2], self.visit_count[i] / self.step_count)

    def __get_fairness(self, values):
        """Calculate Jain's fairness index for a set of values"""
        square_of_sum = np.square(np.sum(values))
        sum_of_square = np.sum(np.square(values))
        if sum_of_square == 0:
            return 0.0
        jain_fairness_index = square_of_sum / (sum_of_square * float(len(values)))
        return jain_fairness_index

    def __get_reward(self, new_visit_count, energy_consumed, fairness, new_positions):
        """Calculate reward"""
        if self.step_count > 1:
            coverage_incr = np.sum((new_visit_count / self.step_count) - (self.visit_count / (self.step_count - 1)))
        else:
            coverage_incr = np.sum(new_visit_count / self.step_count)

        # Calculate the grid entropy 
        grid = np.zeros((self.map_width, self.map_height))
        for x, y in new_positions:
            grid[int(x)][int(y)] += 1
        prob = grid / np.sum(grid)
        entropy = np.sum(prob * np.log(prob + 1e-6))

        return ((fairness * coverage_incr) / (energy_consumed + self.epsilon)) - (self.entropy_factor * entropy)

    def step(self, action_list):
        """Process one step of the environment given agent actions"""
        actions = copy.copy(action_list)
        self.step_count += 1
        reward = [0] * self.num_uavs
        new_positions = []
        self.coverage_map.fill(False)  # Reset coverage map
        new_visit_count = copy.deepcopy(self.visit_count)
        energy_consumed = 0.0  # Total energy consumed in this step

        # Process each UAV's action
        for i in range(self.num_uavs):
            if self.dn[i]:
                new_positions.append(self.uav_pos[i])
                continue

            action = actions[i]
            angle = action[0] * 2 * np.pi  # Map from [0,1] to [0,2π]
            distance_ratio = action[1]  # Map from [0,1] to [0,1]

            distance = distance_ratio * self.max_dist
            delta_x = distance * np.cos(angle)
            delta_y = distance * np.sin(angle)
            new_x = self.uav_pos[i][0] + delta_x
            new_y = self.uav_pos[i][1] + delta_y

            # Check boundary constraints
            if 0 <= new_x < self.map_width and 0 <= new_y < self.map_height:
                new_positions.append([new_x, new_y])
            else:
                # Stay in place and apply wall penalty
                new_positions.append([self.uav_pos[i][0], self.uav_pos[i][1]])
                reward[i] += self.p_wall
                self.penalty[i] += 1

            # Calculate distances to all data points
            _dis_sq = np.sum(np.square(self.datas - new_positions[-1]), axis=1)

            # Cover points within range
            for index, dis_sq in enumerate(_dis_sq):
                if dis_sq <= self.cov_range**2:
                    if not self.coverage_map[index]:
                        self.coverage_map[index] = True
                        new_visit_count[index] += 1

            # Consume energy
            energy_consumed_uav = min(self.energy_factor * distance + self.hover_energy * (1 - distance_ratio), self.energy[i])
            self.energy[i] -= energy_consumed_uav
            energy_consumed += energy_consumed_uav

            # Check if energy is depleted
            if self.energy[i] <= self.epsilon * self.max_energy:
                self.dn[i] = True

        # Check for disconnected UAVs
        for i in range(self.num_uavs):
            is_connected = False
            for j in range(self.num_uavs):
                if j != i:
                    dist = np.linalg.norm(np.array(new_positions[i]) - np.array(new_positions[j]))
                    if dist <= self.comm_range:
                        is_connected = True
                        break

            if not is_connected:
                self.penalty[i] += 1
                reward[i] += self.p_comm

        # Calculate common reward and metrics
        done = sum(self.dn) == self.num_uavs  # Done if all UAVs are depleted
        avg_coverage_score = np.mean(new_visit_count / self.step_count)
        fairness = self.__get_fairness(new_visit_count.astype(np.float64))

        total_energy_consumed = np.sum(self.max_energy - self.energy)  # Cumulative energy
        normalized_energy = total_energy_consumed / (self.num_uavs * self.step_count * self.energy_factor * self.max_dist)
        avg_energy_eff = (fairness * avg_coverage_score) / (normalized_energy + self.epsilon)

        common_reward = self.__get_reward(new_visit_count, energy_consumed, fairness, new_positions)
        for i in range(self.num_uavs):
            if not self.dn[i]:
                reward[i] += common_reward

        # Check for invalid rewards
        for r in reward:
            if np.isnan(r):
                raise ValueError("NaN value detected in reward")

        clear_uav_pos = copy.copy(self.uav_pos)
        self.uav_pos = new_positions
        self.visit_count = new_visit_count
        self.__update_state(clear_uav_pos)

        return (self.state, done, reward, (avg_coverage_score, fairness, avg_energy_eff, self.penalty))

    def reset(self):
        """Reset environment to initial state for a new episode"""
        self.step_count = 0
        self.energy.fill(self.max_energy)
        self.dn = [False] * self.num_uavs
        self.penalty.fill(0)
        self.coverage_map.fill(False)
        self.visit_count.fill(0)
        self.uav_pos = copy.deepcopy(self.init_positions)

        self.__init_state()
        return self.state
