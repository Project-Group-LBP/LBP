from data_points import test_data
import copy
import numpy as np
from gymnasium import spaces
from PIL import Image
import os

# Environment parameters
map_width = map_height = 16
grid_width = grid_height = 80
wall_width = 4
wall_value = -1
channel = 3
num_uavs = 6
init_positions = [[4, 4], [12, 4], [4, 12], [12, 12], [8, 8], [8, 4]]
max_energy = 500
num_action = 2
comm_range = 1.1
max_distance = 1.0
wall_penalty = -1.0
comm_broken_penalty = -1.0
alpha = 1.0
epsilon = 1e-4
factor = 0.1


class Env:
    def __init__(self, image_init=False, log_dir="."):
        self.map_width = map_width
        self.map_height = map_height
        self.width = grid_width
        self.height = grid_height
        self.channels = channel
        self.img_path = log_dir
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        # UAV configuration
        self.num_uavs = num_uavs
        self.observation_space = [spaces.Box(low=-1, high=1, shape=(self.width, self.height, self.channels)) for _ in range(self.num_uavs)]
        self.action_space = [spaces.Box(low=-1, high=1, shape=(num_action,)) for _ in range(self.num_uavs)]

        # Movement and collection parameters
        self.max_energy = max_energy
        self.comm_range = comm_range
        self.max_dist = max_distance
        self.alpha = alpha  # Energy needed per unit data collected
        self.factor = factor  # Energy needed per unit distance moved
        self.epsilon = epsilon  # A small value for reference
        self.step_count = 0
        self.dn = [False] * self.num_uavs  # UAVs with depleted energy
        self.p_wall = wall_penalty
        self.p_comm = comm_broken_penalty
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

        self._init_data_map = np.zeros((self.width, self.height)).astype(np.float16)
        self._init_position_map = np.zeros((num_uavs, self.width, self.height)).astype(np.float16)

        # Draw walls and data points on data map
        self._draw_wall(self._init_data_map)
        for pos in self.datas:
            self._draw_data_point(pos[0], pos[1], self._init_data_map)

        # Draw initial UAV positions
        for i_n in range(self.num_uavs):
            self._draw_UAV(init_positions[i_n][0], init_positions[i_n][1], 1.0, self._init_position_map[i_n])

    def _transform_coords(self, x, y):
        """Transform logical coordinates to visual coordinates"""
        return int(4 * x + wall_width * 2), int(4 * y + wall_width * 2)

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
            for i in range(wall_width):
                grid[i][j] = wall_value
            for i in range(self.height - wall_width, self.height):
                grid[i][j] = wall_value
        for i in range(self.width):
            for j in range(wall_width):
                grid[i][j] = wall_value
            for j in range(self.height - wall_width, self.height):
                grid[i][j] = wall_value

    def _draw_data_point(self, x, y, grid, value=1.0):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 2, 2, value, grid, add=True)

    def _draw_UAV(self, x, y, value, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 4, 4, value, grid)

    def _clear_data_point(self, x, y, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 2, 2, 0, grid)

    def _clear_uav(self, x, y, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 4, 4, 0, grid)

    def save_image(self, name=None, include_uavs=True):
        grid = self.image_data.copy()
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
                            rgb_img[x + dx, y + dy] = [0.3, 1.0, 0.3]
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

    def __init_state(self):
        """Initialize image representation for each UAV"""
        self.image_data = copy.copy(self._init_data_map)
        self.image_position = copy.copy(self._init_position_map)
        self.image_track = np.zeros(self.image_position.shape)

        # Create state representation for each UAV
        state = []
        for i in range(self.num_uavs):
            image = np.zeros((self.width, self.height, self.channels)).astype(np.float16)
            image[:, :, 0] = self.image_data
            image[:, :, 1] = self.image_position[i]
            # image[:, :, 2] is already initialized to zeros
            state.append(image)
        self.state = state

    def __update_state(self, clear_uav):
        """Update state representation after UAV movements and data collection"""
        for n in range(self.num_uavs):
            # Update UAV positions (channel 1)
            self._clear_uav(clear_uav[n][0], clear_uav[n][1], self.state[n][:, :, 1])
            self._draw_UAV(self.uav_pos[n][0], self.uav_pos[n][1], self.energy[n] / self.max_energy, self.state[n][:, :, 1])

            # Update coverage information (channel 2)
            self.state[n][:, :, 2].fill(0.0)
            max_visit = max(1, np.max(self.visit_count))
            for i, pos in enumerate(self.datas):
                if self.visit_count[i] > 0:
                    self._draw_data_point(pos[0], pos[1], self.state[n][:, :, 2], min(self.visit_count[i] / max_visit, 1.0))

    def __get_fairness(self, values):
        """Calculate Jain's fairness index for a set of values"""
        square_of_sum = np.square(np.sum(values))
        sum_of_square = np.sum(np.square(values))
        if sum_of_square == 0:
            return 0.0
        jain_fairness_index = square_of_sum / (sum_of_square * float(len(values)))
        return jain_fairness_index

    def __get_reward(self, new_visit_count, old_visit_count, energy_consumed, fairness):
        """Calculate reward"""
        coverage_incr = np.sum((new_visit_count / self.step_count) - (old_visit_count / (self.step_count - 1)))
        return fairness * coverage_incr / (energy_consumed + self.epsilon)

    def step(self, action_list):
        """Process one step of the environment given agent actions"""
        actions = copy.deepcopy(action_list)
        self.step_count += 1

        # Check for invalid actions
        for i in range(self.num_uavs):
            if np.isnan(actions[i]).any():
                raise ValueError("NaN value detected in action")

        # Initialize step variables
        reward = [0] * self.num_uavs
        clear_uav = copy.copy(self.uav_pos)
        new_positions = []

        self.coverage_map.fill(False)  # Reset coverage map
        new_visit_count = copy.deepcopy(self.visit_count)

        energy_consumed = 0.0

        # Process each UAV's action
        for i in range(self.num_uavs):
            # Record trajectory
            if self.dn[i]:
                new_positions.append(self.uav_pos[i])
                continue

            # Ensure actions are in the correct format
            action = actions[i]

            # Check if action is a nested array and flatten if needed
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action.flatten()

            # Ensure we have exactly 2 values
            if len(action) != 2:
                raise ValueError(f"Expected action to have 2 values, but got {len(action)}")

            # actions[0] is angle in radians (scaled from [-1,1] to [0,2π])
            # actions[1] is distance ratio (scaled from [-1,1] to [0,1])
            angle = (action[0] + 1) * np.pi  # Map from [-1,1] to [0,2π]
            distance_ratio = (action[1] + 1) / 2  # Map from [-1,1] to [0,1]

            distance = distance_ratio * self.max_dist
            # Limit movement based on available energy
            if self.energy[i] < distance:
                distance = distance_ratio * self.energy[i]
            delta_x = int(distance * np.cos(angle))
            delta_y = int(distance * np.sin(angle))

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
            _pos = np.repeat([new_positions[-1]], [self.datas.shape[0]], axis=0)
            _minus = self.datas - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)

            # Process data collection for points within range
            for index, dis in enumerate(_dis):
                if np.sqrt(dis) <= self.comm_range:
                    if not self.coverage_map[index]:
                        self.coverage_map[index] = True
                        new_visit_count[index] += 1

            # Consume energy
            self.energy[i] -= self.factor * distance
            energy_consumed += self.factor * distance

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
        fairness = self.__get_fairness(new_visit_count)

        total_energy_consumed = np.sum(self.max_energy - self.energy)  # Cumulative energy
        normalized_energy = total_energy_consumed / (self.num_uavs * self.step_count * self.max_energy)
        energy_efficiency = (fairness * avg_coverage_score) / (normalized_energy + self.epsilon)

        if self.step_count > 1:
            common_reward = self.__get_reward(new_visit_count, self.visit_count, energy_consumed, fairness)
            for i in range(self.num_uavs):
                reward[i] += common_reward

        # Check for invalid rewards
        for r in reward:
            if np.isnan(r):
                raise ValueError("NaN value detected in reward")

        self.uav_pos = new_positions
        self.visit_count = new_visit_count
        self.__update_state(clear_uav)

        # Return state, done flag, reward, and metrics
        return (self.state, done, reward, (avg_coverage_score, fairness, energy_efficiency, self.penalty))

    def reset(self):
        """Reset environment to initial state for a new episode"""
        self.step_count = 0
        self.coverage_map = np.zeros(self.total_points, dtype=bool)
        self.visit_count = np.zeros(self.total_points, dtype=np.int16)
        self.uav_pos = copy.deepcopy(init_positions)
        self.energy = np.ones(self.num_uavs).astype(np.float64) * self.max_energy
        self.dn = [False] * self.num_uavs
        self.penalty = np.zeros(self.num_uavs)

        self.__init_state()
        return self.state
