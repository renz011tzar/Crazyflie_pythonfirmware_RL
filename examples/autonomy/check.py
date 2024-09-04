import torch

# Load the checkpoint
checkpoint = torch.load('Crazyflie.pth')

# Print all the keys in the checkpoint
for key in checkpoint.keys():
    print(key)

# If you want to dive deeper into the model's state_dict:
if 'model' in checkpoint:
    print("\nKeys in the 'model' state_dict:")
    for key in checkpoint['model'].keys():
        print(key)
