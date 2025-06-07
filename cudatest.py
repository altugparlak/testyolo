import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU