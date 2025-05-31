import numpy
from minesweeper_common import UNKNOWN, MINE, get_neighbors

RUN_TESTS = False


class Player:
    def __init__(self, rows, columns, game, mine_prb):
        # Initialize a player for a game on a board of given size with the probability of a mine on each cell.
        self.rows = rows
        self.columns = columns
        self.mine_prb = mine_prb

        # Matrix of the game. Every cell contains a number with the following meaning:
        # - A non-negative integer for explored cells means the number of mines in neighbors
        # - MINE: A cell is marked that it contains a mine
        # - UNKNOWN: We do not know whether it contains a mine or not.
        self.game = game

        # Matrix which for every cell contains the list of all neighbors.
        self.neighbors = get_neighbors(rows, columns)

        # Matrix of numbers of missing mines in the neighborhood of every cell.
        # -1 if a cell is unexplored.
        self.mines = numpy.full((rows, columns), -1)

        # Matrix of the numbers of unexplored neighborhood cells, excluding known mines.
        self.unknown = numpy.full((rows, columns), 0)
        for i in range(self.rows):
            for j in range(self.columns):
                self.unknown[i,j] = len(self.neighbors[i,j])

        # A set of cells for which the precomputed values self.mines and self.unknown need to be updated.
        self.invalid = set()


    def turn(self):
        # Returns the position of one cell to be explored.
        pos = self.preprocessing()
        if not pos:
            pos = self.probability_player()
        self.invalidate_with_neighbors(pos)
        return pos


    def probability_player(self):
        # Return an unexplored cell with the minimal probability of mine
        prb = self.get_each_mine_probability()
        min_prb = 1
        for i in range(self.rows):
            for j in range(self.columns):
                if self.game[i,j] == UNKNOWN:
                    if prb[i,j] > 0.9999: # Float-point arithmetics may not be exact.
                        self.game[i,j] = MINE
                        self.invalidate_with_neighbors((i,j))
                    if min_prb > prb[i,j]:
                        min_prb = prb[i,j]
                        best_position = (i,j)
        return best_position


    def invalidate_with_neighbors(self, pos):
        # Insert a given position and its neighborhood to the list of cell requiring update of precomputed information.
        self.invalid.add(pos)
        for neigh in self.neighbors[pos]:
            self.invalid.add(neigh)

    def preprocess_all(self):
        # Preprocess all cells
        self.invalid = set((i,j) for i in range(self.rows) for j in range(self.columns))
        pos = self.preprocessing()
        assert(pos == None) # Preprocessing is incomplete


    def preprocessing(self):
        """
        Simple counting inference: for every explored cell, if the number of missing mines
        equals the number of unknown neighbors, flag them all; if zero missing, pick a safe one.
        Returns a single guaranteed‐safe cell to explore, or None if none.
        """
        while self.invalid:
            pos = self.invalid.pop()

            # Count the numbers of unexplored neighborhood cells, excluding known mines.
            self.unknown[pos] = sum(1 if self.game[neigh] == UNKNOWN else 0 for neigh in self.neighbors[pos])

            if self.game[pos] >= 0:
                # If the cell pos is explored, count the number of missing mines in its neighborhood.
                self.mines[pos] = self.game[pos] - sum(1 if self.game[neigh] == MINE else 0 for neigh in self.neighbors[pos])
                assert(0 <= self.mines[pos] and self.mines[pos] <= self.unknown[pos])

                if self.unknown[pos] > 0:
                    if self.mines[pos] == self.unknown[pos]:
                        # All unexplored neighbors have to contain a mine, so mark them.
                        for neigh in self.neighbors[pos]:
                            if self.game[neigh] == UNKNOWN:
                                self.game[neigh] = MINE
                                self.invalidate_with_neighbors(neigh)

                    elif self.mines[pos] == 0:
                        # All mines in the neighborhood was found, so explore the rest.
                        self.invalid.add(pos) # There may be other unexplored neighbors.
                        for neigh in self.neighbors[pos]:
                            if self.game[neigh] == UNKNOWN:
                                return neigh
                        assert(False) # There has to be at least one unexplored neighbor.

        if not RUN_TESTS:
            return None

        # If the invalid list is empty, so self.unknown and self.mines should be correct.
        # Verify it to be sure.
        for i in range(self.rows):
            for j in range(self.columns):
                assert(self.unknown[i,j] == sum(1 if self.game[neigh] == UNKNOWN else 0 for neigh in self.neighbors[i,j]))
                if self.game[i,j] >= 0:
                    assert(self.mines[i,j] == self.game[i,j] - sum(1 if self.game[neigh] == MINE else 0 for neigh in self.neighbors[i,j]))


    def get_each_mine_probability(self):
        """
        Return a matrix of marginal probabilities that each cell contains a mine,
        conditional on the current information.
        We identify the 'frontier' (unknown cells adjacent to at least one number) and
        split it into connected components of constraints.  For each component
        up to ~20 variables we enumerate all mine‐assignments consistent with
        the local counts and weight them by the prior mine probability.  Unconstrained
        cells get a dynamic global estimate.
        """
        rows, cols = self.rows, self.columns
        # --- dynamic global probability based on remaining estimate ---
        total_mines = rows * cols * self.mine_prb
        flagged = numpy.sum(self.game == MINE)
        unknown_total = numpy.sum(self.game == UNKNOWN)
        if unknown_total > 0:
            global_pr = (total_mines - flagged) / unknown_total
            # clamp to [0,1]
            if global_pr < 0: global_pr = 0.0
            if global_pr > 1: global_pr = 1.0
        else:
            global_pr = 0.0
        pr = numpy.full((rows, cols), global_pr)

        # 1) Known flags and explored cells override
        for i in range(rows):
            for j in range(cols):
                if self.game[i,j] == MINE:
                    pr[i,j] = 1.0          # flagged
                elif self.game[i,j] >= 0:
                    pr[i,j] = 0.0          # explored safe

        # 2) Build variables (unknown frontier cells) and constraints (number cells)
        var_idx = {}    # map (r,c) -> variable index
        rev_vars = []   # list: index -> (r,c)
        cons = []       # list of ( [var indices], mines_needed )

        for i in range(rows):
            for j in range(cols):
                if self.game[i,j] >= 0:
                    nbrs = self.neighbors[i,j]
                    unk = [p for p in nbrs if self.game[p] == UNKNOWN]
                    if not unk:
                        continue
                    flagged_nb = sum(1 for p in nbrs if self.game[p] == MINE)
                    needed = self.game[i,j] - flagged_nb
                    idxs = []
                    for p in unk:
                        if p not in var_idx:
                            var_idx[p] = len(rev_vars)
                            rev_vars.append(p)
                        idxs.append(var_idx[p])
                    cons.append((idxs, needed))

        n_vars = len(rev_vars)
        if n_vars == 0:
            return pr

        # 3) Adjacency: which constraints mention each var?
        cons_of_var = [[] for _ in range(n_vars)]
        for c_id, (idxs, _) in enumerate(cons):
            for v in idxs:
                cons_of_var[v].append(c_id)

        visited_var = [False]*n_vars
        visited_cons = [False]*len(cons)
        MAX_ENUM = 20

        # 4) Find connected components and enumerate (or approximate)
        for start in range(n_vars):
            if visited_var[start]:
                continue
            queue = [start]
            comp_vars = []
            comp_cons = []
            visited_var[start] = True
            while queue:
                v = queue.pop()
                comp_vars.append(v)
                for c in cons_of_var[v]:
                    if not visited_cons[c]:
                        visited_cons[c] = True
                        comp_cons.append(c)
                        for v2 in cons[c][0]:
                            if not visited_var[v2]:
                                visited_var[v2] = True
                                queue.append(v2)

            k = len(comp_vars)
            if k > MAX_ENUM:
                # too large: approximate by local neighbor ratios
                for gv in comp_vars:
                    cell = rev_vars[gv]
                    ratios = []
                    for c_id in cons_of_var[gv]:
                        if c_id in comp_cons:
                            idxs, need = cons[c_id]
                            cnt = len(idxs)
                            if cnt > 0:
                                ratios.append(need / cnt)
                    pr[cell] = max(ratios) if ratios else global_pr
                continue

            # Map global var index -> local 0..k-1
            local_index = {gv: li for li, gv in enumerate(comp_vars)}
            local_cons = []
            for c in comp_cons:
                g_idxs, need = cons[c]
                l_idxs = [local_index[g] for g in g_idxs]
                local_cons.append((l_idxs, need))

            total_w = 0.0
            w_sums = [0.0]*k
            for mask in range(1 << k):
                bits = mask.bit_count()
                w = (self.mine_prb**bits) * ((1-self.mine_prb)**(k-bits))
                ok = True
                for l_idxs, need in local_cons:
                    s = sum((mask>>li)&1 for li in l_idxs)
                    if s != need:
                        ok = False
                        break
                if not ok:
                    continue
                total_w += w
                for li in range(k):
                    if (mask >> li) & 1:
                        w_sums[li] += w

            if total_w > 0:
                for li, gv in enumerate(comp_vars):
                    cell = rev_vars[gv]
                    pr[cell] = w_sums[li] / total_w

        return pr
