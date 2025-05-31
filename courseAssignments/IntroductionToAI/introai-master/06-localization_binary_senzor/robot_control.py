import numpy, scipy, networkx  # You can use everything from these libraries if you find them useful.

class RobotControl:
    def __init__(self, environment):
        self.env = environment

        # Map organizing and station position
        self.rows = environment.rows
        self.columns = environment.columns
        self.dest_i, self.dest_j = map(int, environment.destination)

        # Precompute the best first step from each cell
        # toward the station by Manhattan distance
        self.dist_map = numpy.zeros((self.rows, self.columns), dtype=float)
        for i in range(self.rows):
            for j in range(self.columns):
                self.dist_map[i, j] = abs(i - self.dest_i) + abs(j - self.dest_j)

        # Precompute the best first step from each cell
        # toward the station by Manhattan distance
        directions = {
            self.env.NORTH: (-1, 0),
            self.env.EAST:  (0, 1),
            self.env.SOUTH: (1, 0),
            self.env.WEST:  (0, -1),
        }

        self.moves = directions
        self.goto_dir = numpy.zeros((self.rows, self.columns), dtype=int)
        
        for i in range(self.rows):
            for j in range(self.columns):
                best = self.env.NORTH
                best_dist = self.dist_map[i, j]
                for d, (di, dj) in self.moves.items():
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.columns:
                        if self.dist_map[ni, nj] < best_dist:
                            best_dist = self.dist_map[ni, nj]
                            best = d
                # if already at dest, just pick any
                self.goto_dir[i, j] = best

        # Sensor model: grayscale probability and its complement
        self.grayscale = environment.grayscale
        self.grayscale_complement = 1 - self.grayscale

        # Uniform prior belief over all cells
        N = self.rows * self.columns
        self.position_dist = numpy.full((self.rows, self.columns), 1.0 / N)

        # Counts steps for battery
        self.total_steps = environment.steps
        self.remaining_steps = self.total_steps

        # Thresholds and weights
        self.belief_threshold = 0.6  # switch to exploit when peak belief > threshold
        self.battery_margin   = (self.rows + self.columns) // 4  # extra buffer steps
        self.safety_thresh    = 0.25  # maximum allowed fall-off probability during explore
        self.lambda_          = 0.6   # trade-off weight for expected-distance vs fall risk

    def normalize_position_distribution(self):
        # Zero out the station cell (we don't return to it) and re-normalize
        self.position_dist[self.dest_i, self.dest_j] = 0.0
        total = float(self.position_dist.sum())
        if total > 0:
            self.position_dist /= total

    def update_position_by_sensor_reading(self, reading):
        # multiply by sensor likelihoods
        if reading:
            self.position_dist *= self.grayscale
        else:
            self.position_dist *= self.grayscale_complement

    def update_position_by_command(self, direction):
        # predict step: convolve belief with perfect motion
        di, dj = self.moves[direction]
        new_dist = numpy.zeros_like(self.position_dist)
        for i in range(self.rows):
            for j in range(self.columns):
                p = self.position_dist[i, j]
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.columns:
                    new_dist[ni, nj] += p
        self.position_dist = new_dist

    def get_probabilities_fall(self):
        # for each direction, compute total belief that would fall off
        fall_probs = [0.0] * 4
        for d, (di, dj) in self.moves.items():
            tot = 0.0
            for i in range(self.rows):
                for j in range(self.columns):
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < self.rows and 0 <= nj < self.columns):
                        tot += self.position_dist[i, j]
            fall_probs[d] = tot
        return fall_probs

    def get_command(self, sensor_reading):
        # sense and update belief
        self.update_position_by_sensor_reading(sensor_reading)
        self.normalize_position_distribution()

        # compute best-guess cell, its probability, distance, and fall risks
        best_i = best_j = 0
        best_p = 0.0
        for i in range(self.rows):
            for j in range(self.columns):
                p = self.position_dist[i, j]
                if p > best_p:
                    best_p = p
                    best_i, best_j = i, j
        dist_to_goal = self.dist_map[best_i, best_j]
        fall_probs = self.get_probabilities_fall()

        # rush mode: if battery is low relative to Manhattan distance
        if self.remaining_steps <= dist_to_goal + self.battery_margin:
            cmd = self.goto_dir[best_i, best_j]
            self.update_position_by_command(cmd)
            self.remaining_steps -= 1
            return cmd

        # exploitation mode: if well-localized by peak belief
        if best_p >= self.belief_threshold:
            # move greedily towards station but choose order (horiz vs vert) by larger offset
            di = self.dest_i - best_i
            dj = self.dest_j - best_j
            cands = []
            if abs(di) >= abs(dj):
                if di > 0: cands.append(self.env.SOUTH)
                elif di < 0: cands.append(self.env.NORTH)
                if dj > 0: cands.append(self.env.EAST)
                elif dj < 0: cands.append(self.env.WEST)
            else:
                if dj > 0: cands.append(self.env.EAST)
                elif dj < 0: cands.append(self.env.WEST)
                if di > 0: cands.append(self.env.SOUTH)
                elif di < 0: cands.append(self.env.NORTH)

            # score each candidate by expected next-step distance + fall risk
            best_score = float('inf')
            best_dir = cands[0] if cands else self.env.NORTH
            for d in cands:
                risk = fall_probs[d]
                di2, dj2 = self.moves[d]
                expd = 0.0
                for i in range(self.rows):
                    for j in range(self.columns):
                        p = self.position_dist[i, j]
                        if p < 1e-8: continue
                        ni, nj = i + di2, j + dj2
                        if 0 <= ni < self.rows and 0 <= nj < self.columns:
                            expd += p * self.dist_map[ni, nj]
                        else:
                            expd += p * (self.rows + self.columns)
                score = self.lambda_ * expd + (1 - self.lambda_) * risk
                if score < best_score:
                    best_score, best_dir = score, d
            cmd = best_dir
            self.update_position_by_command(cmd)
            self.remaining_steps -= 1
            return cmd

        # exploration mode: safe one-step lookahead over all directions
        cands = [d for d in range(4) if fall_probs[d] < self.safety_thresh]
        if not cands:
            cands = list(self.moves.keys())
        best_score = float('inf')
        best_dir = cands[0]
        for d in cands:
            di2, dj2 = self.moves[d]
            expd = 0.0
            for i in range(self.rows):
                for j in range(self.columns):
                    p = self.position_dist[i, j]
                    if p < 1e-8: continue
                    ni, nj = i + di2, j + dj2
                    if 0 <= ni < self.rows and 0 <= nj < self.columns:
                        expd += p * self.dist_map[ni, nj]
                    else:
                        expd += p * (self.rows + self.columns)
            score = self.lambda_ * expd + (1 - self.lambda_) * fall_probs[d]
            if score < best_score:
                best_score, best_dir = score, d
        cmd = best_dir
        self.update_position_by_command(cmd)
        self.remaining_steps -= 1
        return cmd

    def calculate_position_distribution(self, sensor_readings, commands):
        # exactly the same filtering and prediction steps for offline tests
        for idx, cmd in enumerate(commands):
            self.update_position_by_sensor_reading(sensor_readings[idx])
            self.normalize_position_distribution()
            self.update_position_by_command(cmd)
        self.update_position_by_sensor_reading(sensor_readings[-1])
        self.normalize_position_distribution()
        return (self.position_dist, self.get_probabilities_fall())


