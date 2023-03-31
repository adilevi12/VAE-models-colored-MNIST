from hw2_319003323_train import get_data, ContinuousVAE, plot_reconstructed,  DiscreteVAE, reconstruct_grid_discrete, JointVAE, reconstruct_grid_joint
import torch

train_loader, test_loader = get_data()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

continuous_VAE = ContinuousVAE(2).to(device)
continuous = torch.load('continuous_vae.pkl')
continuous_VAE.load_state_dict(continuous)
plot_reconstructed(continuous_VAE)

discrete_VAE = DiscreteVAE(2, 10).to(device)
discrete = torch.load('discrete_vae.pkl')
discrete_VAE.load_state_dict(discrete)
reconstruct_grid_discrete(discrete_VAE, 2, 10)

joint_VAE = JointVAE(2, 2, 10).to(device)
joint = torch.load('joint_vae.pkl')
joint_VAE.load_state_dict(joint)
reconstruct_grid_joint(joint_VAE)