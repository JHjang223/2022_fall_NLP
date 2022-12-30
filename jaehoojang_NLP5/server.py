#setting on yout trion workspace

import os
os.system("CUDA_VISIBLE_DEVICES=0,1 /opt/tritonserver/bin/tritonserver --log-warning false --model-repository=./triton-model-store_hoo/gpt/ &")