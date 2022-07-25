import torch
import os

def set_gpu_env():
    os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

set_gpu_env()

H,W = 64,64
NLINES = 100 #How many lines of data to use for training?
NLTEST = 60 #How many lines for the test set?

N_VALID = 805 #How much to reserve for validation

EPOCHS = 24
NGPUS = torch.cuda.device_count()
BATCH_SIZE = NGPUS * 32
LR = 1e-3 * NGPUS
print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)
NPREDS = 10

drop = 0.1
nconv = 16