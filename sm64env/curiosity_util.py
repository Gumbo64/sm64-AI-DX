import numpy as np

class CURIOSITY:
    def __init__(self, max_visits=1000):
        self.max_visits = max_visits
        self.chunk_xz_size = 40
        self.visit_decay = 0.5
        self.chunk_y_size = 100
        # self.chunk_y_size = 200
        
        self.bounding_size = 8192
        self.radius = 200

        self.F_shape = np.array([2 * self.bounding_size // self.chunk_xz_size, 2 * self.bounding_size // self.chunk_y_size, 2 * self.bounding_size // self.chunk_xz_size])
 
        self.sphere_mask, self.sphere_values = self.create_mask()

        self.F = np.zeros(shape=self.F_shape, dtype=float)
    
    def reset(self):
        self.F = np.zeros(shape=self.F_shape, dtype=float)
    
    def soft_reset(self):
        self.F = self.F * self.visit_decay

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

    def add_circle(self,centre):
        centre = np.array(centre)
        indices = self.sphere_mask.copy()
        values = self.sphere_values.copy()

        indices += self.pos_to_index(centre)
        # don't go over or under the bounds
        indices_indices = ((indices) >= 0).all(axis=1) & ((indices) < np.array(self.F_shape)).all(axis=1)
        values = values[indices_indices]
        indices = indices[indices_indices]

        # self.F[indices[:, 0], indices[:, 1], indices[:, 2]] += values
        self.F[indices[:, 0], indices[:, 1], indices[:, 2]] += 1
        self.F[indices[:, 0], indices[:, 1], indices[:, 2]] = np.clip(self.F[indices[:, 0], indices[:, 1], indices[:, 2]], 0, self.max_visits)
        # self.F[indices] += 1

    def add_circles(self,centres):
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

        indices_indices = ((indices) >= 0).all(axis=1) & ((indices) < np.array(self.F_shape)).all(axis=1)
        indices = indices[indices_indices]
        if len(indices) == 0:
            return
        self.F[indices[:, 0], indices[:, 1], indices[:, 2]] += 1
        self.F[indices[:, 0], indices[:, 1], indices[:, 2]] = np.clip(self.F[indices[:, 0], indices[:, 1], indices[:, 2]], 0, self.max_visits)



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

    def get_visits(self, pos):
        indices = self.pos_to_index(pos)
        return self.F[indices[0], indices[1], indices[2]]
    
    def get_visits_multi(self, positions):
        indices = self.multi_pos_to_index(positions)
        return self.F[indices[:, 0], indices[:, 1], indices[:, 2]]


    # Get the maximum number of visits in a sphere around the position
    def get_sphere_visits(self, pos):
        sphere = self.sphere_mask + self.pos_to_index(pos)
        sphere = sphere[((sphere) >= 0).all(axis=1) & ((sphere) < np.array(self.F_shape)).all(axis=1)]
        visits = self.F[sphere[:, 0], sphere[:, 1], sphere[:, 2]]
        return np.max(visits)

    
    def get_sphere_visits_multi(self, positions):
        return np.array([self.get_sphere_visits(pos) for pos in positions])


    def entropy(self):
        non_zero = self.F[self.F > 0].flatten()
        if len(non_zero) == 0:
            return np.inf
        
        non_zero = non_zero / np.sum(non_zero)
        entropy = -np.sum(non_zero * np.log(non_zero))
        terms = len(non_zero)
        return  entropy, terms
        


