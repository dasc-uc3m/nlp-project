import torch
if torch.backends.mps.is_available():
    print("MPS is available!")
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("Test tensor on MPS:", x)
else:
    print("MPS is not available.")
# Check if usable
print(f"Is MPS built? {torch.backends.mps.is_built()}")
