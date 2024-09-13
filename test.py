def data_augmentation(returns, log_probs, values, obs_s, actions, advantages, num_augmentation_factor):
    # Also need to do it for values, everything


    # Data augmentation: Rotate the position and velocity of each token around the y-axis
    new_obs_s = [obs.clone().to(device) for obs in obs_s] * num_augmentation_factor

    new_actions = actions.repeat(num_augmentation_factor, 1).to(device)
    new_returns = returns.repeat(num_augmentation_factor, 1).to(device)
    new_log_probs = log_probs.repeat(num_augmentation_factor, 1).to(device)
    new_values = values.repeat(num_augmentation_factor, 1).to(device)
    new_advantages = advantages.repeat(num_augmentation_factor, 1).to(device)
    
    # Generate n random angles between 0 and 2*pi
    angles = 2 * torch.pi * torch.rand(num_augmentation_factor)
    
    # Compute cosine and sine of the angles
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Create rotation matrices around the Y-axis
    rotation_matrices_3d = torch.stack([
        cos_angles,  torch.zeros(num_augmentation_factor),  sin_angles,
        torch.zeros(num_augmentation_factor),  torch.ones(num_augmentation_factor),  torch.zeros(num_augmentation_factor),
        -sin_angles,  torch.zeros(num_augmentation_factor),  cos_angles
    ], dim=1).reshape(num_augmentation_factor, 3, 3).to(device)
    rotation_matrics_2d = torch.stack([
        cos_angles, -sin_angles,
        sin_angles, cos_angles
    ], dim=1).reshape(num_augmentation_factor, 2, 2).to(device)

    # -1 so that
    for i in range(0, num_augmentation_factor - 1):
        start_idx = i * len(obs_s)
        end_idx = (i + 1) * len(obs_s)

        for j in range(start_idx, end_idx):
            # Position
            new_obs_s[j][3:6] = torch.matmul(rotation_matrices_3d[i], new_obs_s[j][3:6])
            # Velocity/normal
            new_obs_s[j][6:9] = torch.matmul(rotation_matrices_3d[i], new_obs_s[j][6:9])
            # Rotate stick actions
            new_actions[j][0:2] = torch.matmul(rotation_matrics_2d[i], new_actions[j][0:2])

        # Calculate log probs for the rotated stick actions
        dist, _ = agent.forward([new_obs_s[j] for j in range(start_idx, end_idx)])
        new_log_probs[start_idx:end_idx] = dist.log_prob(new_actions[start_idx:end_idx])

    return new_returns, new_log_probs, new_values, new_obs_s, new_actions, new_advantages

# Data augmentation: Rotate the position and velocity of each token around the y-axis
num_augmentation_factor = 2
with torch.no_grad():
    returns, log_probs, values, obs_s, actions, advantages = data_augmentation(
        returns, log_probs, values, obs_s, actions, advantages, num_augmentation_factor)