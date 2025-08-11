import torch

def check_memory(step):
    print(f"{step} - Torch Memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MiB")
    print(f"{step} - Torch Memory reserved: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MiB")
    #print(f"{step} - Cupy Memory used (allocated): {cp.get_default_memory_pool().used_bytes() / (1024 * 1024):.2f} MiB")
    #print(f"{step} - Cupy Memory reserved: {cp.get_default_memory_pool().total_bytes() / (1024 * 1024):.2f} MiB")
    print(f"{step} - Free Memory: {torch.cuda.mem_get_info()[0] / (1024 * 1024):.2f} MiB")
