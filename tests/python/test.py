import torch
# Create a CUDA stream
stream = torch.cuda.Stream()
stream_handle = stream.cuda_stream
print(type(stream))