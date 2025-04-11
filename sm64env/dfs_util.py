import numpy as np

class DFS_UTIL:
    def __init__(self):
        self.chunk_xz_size = 40
        self.chunk_y_size = 100
        # self.chunk_y_size = 200
        
        self.bounding_size = 8192
        self.radius = 200

        self.F_shape = np.array([2 * self.bounding_size // self.chunk_xz_size, 2 * self.bounding_size // self.chunk_y_size, 2 * self.bounding_size // self.chunk_xz_size])
 
        self.sphere_mask, self.sphere_values = self.create_mask()

        self.F_time = np.full(shape=self.F_shape, fill_value=np.inf, dtype=float)
        self.F_id = np.full(shape=self.F_shape, fill_value=np.inf, dtype=float)

        self.id_counter = 0

    

    def create_ellipsoid_tensor(self, shape_sphere):
        a, b, c = shape_sphere
        x = np.linspace(-1, 1, a)
        y = np.linspace(-1, 1, b)
        z = np.linspace(-1, 1, c)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

        ellipsoid = (xv)**2 + (yv)**2 + (zv)**2 <= 1
        values = np.exp(-(xv**2 + yv**2 + zv**2))
        return ellipsoid, values

    def create_mask(self):
        shape_sphere = np.array([(2 * self.radius) // self.chunk_xz_size + 1, (2 * self.radius) // self.chunk_y_size + 1, (2 * self.radius) // self.chunk_xz_size + 1])
        # print(shape_rad)
        tensor, values = self.create_ellipsoid_tensor(shape_sphere)

        indices = np.argwhere(tensor)
        values = values[indices[:, 0], indices[:, 1], indices[:, 2]]

        indices -= shape_sphere//2
        return indices, values

    def add_circle(self, centre, timestamp, old_centre=None):
        centre = np.array(centre)
        indices = self.sphere_mask.copy()

        indices += self.pos_to_index(centre)


        if old_centre is None:
            ids = np.full(len(self.sphere_mask), -1, dtype=int)
        else:
            ids = np.repeat(self.calc_id(old_centre), len(self.sphere_mask))

        # don't go over or under the bounds
        indices_indices = ((indices) >= 0).all(axis=1) & ((indices) < np.array(self.F_shape)).all(axis=1)
        ids = ids[indices_indices]
        indices = indices[indices_indices]

        # Only write over it when the time is improved
        indices_indices = self.F_time[indices[:, 0], indices[:, 1], indices[:, 2]] > timestamp
        ids = ids[indices_indices]
        indices = indices[indices_indices]

        self.F_time[indices[:, 0], indices[:, 1], indices[:, 2]] = timestamp
        self.F_id[indices[:, 0], indices[:, 1], indices[:, 2]] = ids

    def add_circles(self, centres, timestamp, old_centres=None):
        if len(centres) == 0:
            return
        # for centre in centres:
        #     self.add_circle(centre)
        centre_indices = self.multi_pos_to_index(centres)
        sphere_indices = self.sphere_mask
        
        # Reshape A to (n, 1, 3) and B to (1, m, 3) for broadcasting
        A_expanded = centre_indices[:, np.newaxis, :]
        B_expanded = sphere_indices[np.newaxis, :, :]
        
        # Calculate all combinations using broadcasting
        combinations = A_expanded + B_expanded
        indices = combinations.reshape(-1, 3)

        if old_centres is None:
            ids = np.full(len(centres), np.inf, dtype=int)
        else:
            ids = self.calc_id_multi(old_centres)
        ids = np.repeat(ids, len(self.sphere_mask))

        indices_indices = ((indices) >= 0).all(axis=1) & ((indices) < np.array(self.F_shape)).all(axis=1)
        ids = ids[indices_indices]
        indices = indices[indices_indices]

        indices_indices = self.F_time[indices[:, 0], indices[:, 1], indices[:, 2]] > timestamp
        ids = ids[indices_indices]
        indices = indices[indices_indices]


        if len(indices) == 0:
            return
        self.F_time[indices[:, 0], indices[:, 1], indices[:, 2]] = np.minimum(self.F_time[indices[:, 0], indices[:, 1], indices[:, 2]], timestamp)
        self.F_id[indices[:, 0], indices[:, 1], indices[:, 2]] = ids
        

    def pos_to_index(self,pos):
        x = np.clip((pos[0] + self.bounding_size) // self.chunk_xz_size, 0, self.F_shape[0]-1)
        y = np.clip((pos[1] + self.bounding_size) // self.chunk_y_size, 0, self.F_shape[1]-1)
        z = np.clip((pos[2] + self.bounding_size) // self.chunk_xz_size, 0, self.F_shape[2]-1)
        return (int(x),int(y),int(z))
    
    def index_to_pos(self,index):
        x = index[0] * self.chunk_xz_size - self.bounding_size
        y = index[1] * self.chunk_y_size - self.bounding_size
        z = index[2] * self.chunk_xz_size - self.bounding_size
        return np.array([x,y,z])
    
    def multi_pos_to_index(self, positions):
        indices = np.zeros_like(positions, dtype=int)
        indices[:, 0] = np.clip((positions[:, 0] + self.bounding_size) // self.chunk_xz_size, 0, self.F_shape[0]-1)
        indices[:, 1] = np.clip((positions[:, 1] + self.bounding_size) // self.chunk_y_size, 0, self.F_shape[1]-1)
        indices[:, 2] = np.clip((positions[:, 2] + self.bounding_size) // self.chunk_xz_size, 0, self.F_shape[2]-1)
        return indices
    
    def multi_index_to_pos(self,index):
        x = index[:, 0] * self.chunk_xz_size - self.bounding_size
        y = index[:, 1] * self.chunk_y_size - self.bounding_size
        z = index[:, 2] * self.chunk_xz_size - self.bounding_size
        return np.array([x, y, z]).T

    def get_time(self, pos):
        indices = self.pos_to_index(pos)
        return self.F_time[indices[0], indices[1], indices[2]]
    
    def get_time_multi(self, positions):
        indices = self.multi_pos_to_index(positions)
        return self.F_time[indices[:, 0], indices[:, 1], indices[:, 2]]


    def get_prev_id(self, pos):
        indices = self.pos_to_index(pos)
        return self.F_id[indices[0], indices[1], indices[2]]
    
    def get_prev_id_multi(self, positions):
        indices = self.multi_pos_to_index(positions)
        return self.F_id[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    def calc_id(self, pos):
        indices = self.pos_to_index(pos)
        return indices[0] * self.F_shape[1] * self.F_shape[2] + indices[1] * self.F_shape[2] + indices[2]

    def calc_id_multi(self, positions):
        indices = self.multi_pos_to_index(positions)
        return indices[:, 0] * self.F_shape[1] * self.F_shape[2] + indices[:, 1] * self.F_shape[2] + indices[:, 2]

