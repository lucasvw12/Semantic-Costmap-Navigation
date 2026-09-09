import numpy as np


class OccupancyGrid:
    def __init__(self, height, width, n_classes=1, p_min=0.03, p_max=0.97,
                 label_decay=0.9, label_min_score=0.5):
        self.height = height
        self.width = width
        self.n_classes = n_classes
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.class_scores = np.zeros((n_classes, height, width), dtype=np.float32)
        self.l_min = self.probability_to_logodds(p_min)
        self.l_max = self.probability_to_logodds(p_max)
        self.p_min = p_min
        self.p_max = p_max
        self.label_decay = label_decay
        self.label_min_score = label_min_score

    def probability_to_logodds(self, p):
        p = np.asanyarray(p, dtype=np.float32)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be in the range [0, 1]")
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def logodds_to_probability(self, l):
        return 1 / (1 + np.exp(-l))

    def confidence_to_probability(self, confidence):
        confidence = np.clip(confidence, 0.0, 1.0)
        return 0.5 + 0.45 * confidence

    def collapse(self, rows, cols, confidence):
        valid = (
            (rows >= 0) &
            (rows < self.height) &
            (cols >= 0) &
            (cols < self.width)
        )

        rows = rows[valid]
        cols = cols[valid]
        confidence = confidence[valid]

        flat = rows * self.width + cols
        flat, indices = np.unique(flat, return_index=True)

        return flat // self.width, flat % self.width, confidence[indices]

    def update_cells(self, rows, cols, confidence, class_id=None):
        rows, cols, confidence = self.collapse(rows, cols, confidence)
        if len(rows) == 0:
            return

        log_odds = self.probability_to_logodds(
            self.confidence_to_probability(confidence)
        )

        self.grid[rows, cols] += log_odds
        np.clip(self.grid, self.l_min, self.l_max, out=self.grid)

        if class_id is not None and 0 <= class_id < self.n_classes: 
            self.class_scores[class_id, rows, cols] += confidence

    def update_free_cells(self, rows, cols, confidence, protect=None):
        rows, cols, confidence = self.collapse(rows, cols, confidence)
        if len(rows) == 0:
            return

        if protect is not None:
            keep = ~protect[rows, cols]
            rows = rows[keep]
            cols = cols[keep]
            confidence = confidence[keep]
            if len(rows) == 0:
                return

        log_odds_free = self.probability_to_logodds(
            self.confidence_to_probability(confidence)
        )

        self.grid[rows, cols] -= log_odds_free
        np.clip(self.grid, self.l_min, self.l_max, out=self.grid)

    def get_probability_grid(self):
        return 1 / (1 + np.exp(-self.grid))

    def decay_class_scores(self):
        self.class_scores *= self.label_decay

    def get_label_grid(self): 
        labels = np.full((self.height, self.width), -1, dtype=np.int16)
        settled = self.labelled_mask()
        labels[settled] = np.argmax(self.class_scores, axis=0)[settled]
        return labels

    def labelled_mask(self):
        return self.class_scores.max(axis=0) >= self.label_min_score


class Mapping:
    def __init__(self, cell_size, x_min, y_min, height=None, width=None):
        self.cell_size = cell_size
        self.x_min = x_min
        self.y_min = y_min
        self.height = height
        self.width = width

    def world_to_grid(self, *, x, y):
        col = np.floor((x - self.x_min) / self.cell_size).astype(int)
        row = np.floor((y - self.y_min) / self.cell_size).astype(int)
        return row, col

    def contains(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    def grid_to_world(self, *, row, col):
        x = self.x_min + (np.asarray(col) + 0.5) * self.cell_size
        y = self.y_min + (np.asarray(row) + 0.5) * self.cell_size
        return x, y

    # def find_free_cells(self, start, end):
    #     x0 = int(np.floor((start[0] - self.x_min) / self.cell_size))
    #     y0 = int(np.floor((start[1] - self.y_min) / self.cell_size))
    #     x1 = int(np.floor((end[0] - self.x_min) / self.cell_size))
    #     y1 = int(np.floor((end[1] - self.y_min) / self.cell_size))

    #     cells = []
    #     row = y0
    #     col = x0

    #     sx = 1 if x1 > x0 else -1
    #     sy = 1 if y1 > y0 else -1

    #     dx = abs(y1 - row)
    #     dy = abs(x1 - col)
    #     err = dy - dx

    #     while True:
    #         cells.append((row, col))
    #         if row == y1 and col == x1:
    #             break
    #         err2 = 2 * err
    #         if err2 > -dx:
    #             err -= dx
    #             col += sx
    #         if err2 < dy:
    #             err += dy
    #             row += sy

    #     return np.array(cells).reshape(-1, 2)
