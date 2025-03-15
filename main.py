import ray
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import pynvml
import time
from typing import List, Dict
import matplotlib.pyplot as plt

# Initialize Ray
ray.init(address="auto", ignore_reinit_error=True)

# Global configuration
CUTOFF_MEMORY = 1e9  # 1GB
NUM_JOBS = 3
WORKERS_PER_JOB = 2
GPU_PER_WORKER = 0.1
MAX_EPOCHS = 10
MASTER_ADDR = "localhost"

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_available_gpus() -> Dict[int, int]:
    """Check available GPUs with memory above cutoff"""
    pynvml.nvmlInit()
    gpu_info = {}
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info[i] = info.free
            print(f"GPU {i}: Free memory {info.free/1e9:.2f}GB")
        return {gpu: mem for gpu, mem in gpu_info.items() if mem >= CUTOFF_MEMORY}
    finally:
        pynvml.nvmlShutdown()

@ray.remote(num_gpus=GPU_PER_WORKER)
def train_worker(rank: int, world_size: int, gpu_id: int, job_id: int):
    os.makedirs(f'results/job{job_id}', exist_ok=True)

    """Training worker using DDP"""
    # Set up distributed training
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(12355 + job_id)  # Unique port for each job
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Print environment info for debugging
    print(f"Worker {rank} in job {job_id} using GPU {gpu_id}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    # Initialize the process group
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # When CUDA_VISIBLE_DEVICES is set, the device is remapped to index 0
        torch.cuda.set_device(0)  # Use device 0 since CUDA_VISIBLE_DEVICES remaps the GPU
        
        # Create model and wrap with DDP
        model = Autoencoder().cuda()
        ddp_model = DDP(model, device_ids=[0])  # Use device 0 for DDP as well
        
        # Dataset and dataloader
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ddp_model.parameters(), lr=1e-4)
        
        # Training loop
        for epoch in range(MAX_EPOCHS):
            sampler.set_epoch(epoch)
            for batch_idx, (data, _) in enumerate(loader):
                inputs = data.view(-1, 784).cuda()
                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                # shape of inputs: (64, 784)
                # shape of outputs: (64, 784)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
            
            if rank == 0:
                print(f'Job {job_id} Epoch {epoch} Loss: {loss.item():.4f}')
        
                # plot predictions and inputs side by side
                plt.figure(figsize=(5, 20))
                arr = torch.cat((inputs.detach().cpu().view(-1, 28), outputs.detach().cpu().view(-1, 28)), dim=1)
                plt.imshow(arr)
                plt.savefig(f'results/job{job_id}/predictions_job{job_id}_epoch{epoch}.png')
                plt.close()
                # Save model
                torch.save(ddp_model.state_dict(), f'results/job{job_id}/model_job{job_id}.pth')
                print(f"Model saved to results/job{job_id}/model_job{job_id}.pth")

        dist.destroy_process_group()
        return f"Job {job_id} Worker {rank} completed"
    except Exception as e:
        print(f"Error in worker {rank} of job {job_id}: {str(e)}")
        # Try to clean up
        try:
            dist.destroy_process_group()
        except:
            pass
        raise e

def submit_job(job_id: int):
    """Submit a job with multiple workers to available GPUs"""
    used_gpus = set()  # Track used GPUs for this job
    workers = []
    
    for worker_rank in range(WORKERS_PER_JOB):
        # Find available GPUs not used by any job
        available_gpus = get_available_gpus()
        print(f"Available GPUs: {available_gpus}")
        valid_gpus = [g for g in available_gpus if g not in used_gpus]
        print(f"Valid GPUs for selection: {valid_gpus}")
        
        if not valid_gpus:
            # Clean up any workers already created for this job
            if workers:
                ray.cancel(workers, force=True)
            raise RuntimeError(f"Not enough available GPUs. Used GPUs: {used_gpus}")
            
        # Select GPU with most free memory
        selected_gpu = max(valid_gpus, key=lambda x: available_gpus[x])
        used_gpus.add(selected_gpu)
        print(f"Selected GPU {selected_gpu} for job {job_id}, worker {worker_rank}")
        
        # Submit worker
        workers.append(
            train_worker.remote(
                rank=worker_rank,
                world_size=WORKERS_PER_JOB,
                gpu_id=selected_gpu,
                job_id=job_id
            )
        )
    return workers

# Main execution
if __name__ == "__main__":
    
    # Check how many GPUs are available
    available_gpus = get_available_gpus()
    n_gpus = len(available_gpus)
    print(f"Found {n_gpus} available GPUs")
    
    # Calculate how many jobs we can run
    max_jobs = n_gpus // GPU_PER_WORKER // WORKERS_PER_JOB
    if max_jobs < 1:
        print(f"Not enough GPUs for a single job (need {WORKERS_PER_JOB}, have {n_gpus})")
        exit(1)
    
    jobs_to_run = min(NUM_JOBS, max_jobs)
    print(f"Will run {jobs_to_run} jobs (each requiring {WORKERS_PER_JOB} GPUs)")
    
    # Submit multiple jobs
    all_workers = []
    for job_id in range(jobs_to_run):
        print(f"Submitting job {job_id}")
        try:
            job_workers = submit_job(job_id)
            all_workers.extend(job_workers)
        except Exception as e:
            print(f"Failed to submit job {job_id}: {str(e)}")
            break
    
    # Wait for all jobs to complete
    if all_workers:
        try:
            results = ray.get(all_workers)
            for res in results:
                print(res)
        except Exception as e:
            print(f"Error during job execution: {str(e)}")
    else:
        print("No workers were successfully submitted.")
