from PIL import Image
import os
import os.path
import numpy as np
import torch

def fast_triplets(n, k, gamma=1):
    # https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    # a, b = n_choose_2_elements(n - k)

    a, b = np.triu_indices(n - k*gamma,1) # all combinations of n choose 2 from 

    triplets = np.stack([a, a, b + k*gamma], axis=0)
    triplets = triplets.repeat(k, axis=1)
    triplets[1] += np.tile(np.arange(1, k+1), len(a))
    return triplets.T



def default_image_loader(path):
    return Image.open(path).convert('RGB')

# https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_image_loader.py loosely off this
class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, k=5, gamma=1, transform=None,
                 loader=default_image_loader):
        self.k = k # decides the number of positive examples in front of the anchor
        self.gamma = gamma #decides how large the gap is between positive examples and negative examples, gamma=1 means no gap

        # Scan data directory
        self.base_path = base_path
        self.game_names = []
        self.name_to_id = {}

        self.triplet_game_starts = {}
        self.triplet_game_lengths = {}

        self.game_lengths = {}
        


        game_id_i = 0
        for game_name in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, game_name)):
                self.game_names.append(game_name)
                self.name_to_id[game_name] = game_id_i
                game_id_i += 1

                folder_files = os.listdir(os.path.join(base_path, game_name))
                self.game_lengths[game_name] = len([f for f in folder_files if f.endswith('.png')])

        
        # Create triplets for each game
        self.triplets = np.empty((0, 4), dtype=int)

        current_triplet_index = 0
        for game_name in self.game_names:
            n = self.game_lengths[game_name]
            new_triplets = fast_triplets(n, self.k, self.gamma)

            self.triplet_game_starts[game_name] = current_triplet_index
            self.triplet_game_lengths[game_name] = len(new_triplets)
            current_triplet_index += len(new_triplets)

            # add the id for the game
            new_triplets = np.hstack((np.full((len(new_triplets), 1), self.name_to_id[game_name]), new_triplets))

            self.triplets = np.vstack((self.triplets, new_triplets))
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        game, num1, num2, num3 = self.triplets[index]
        img1 = self.loader(os.path.join(self.base_path, self.game_names[game], f'{num1}.png')) 
        img2 = self.loader(os.path.join(self.base_path, self.game_names[game], f'{num2}.png'))
        img3 = self.loader(os.path.join(self.base_path, self.game_names[game], f'{num3}.png'))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)





