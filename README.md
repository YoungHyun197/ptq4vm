# [WACV 2025] PTQ4VM: Post-training Quantization for Visual Mamba

This is official code for the paper [PTQ4VM].


## How to use
* Generate activation smoothing scale  
```
torchrun --nproc_per_node 1 generate_act_scale.py --resume [model-path] --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path [imagenet path] --batch-size 256
```

+ PTQ4VM
```
torchrun --nproc_per_node 1 quant.py --eval --resume [model-path] --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path [imagenet-path] --act_scales [smoothing-path] --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 16 --n-lvw 16 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2
```
+ CUDA Kernel measurement
1. Install the CUDA Kernel
```
python ./cuda_measure/setup_vim_GEMM.py install
```

2. Check the layer-wise acceleration
```
python cuda_sandbox.py
```

3. Check the end-to-end acceleration
```
torchrun --nproc_per_node 1 quant.py --eval --time_compare --resume [model-path] --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path [imagenet-path] --act_scales [smoothing-path] --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 16 --n-lvw 16 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2
```

For experimental details and hyper-paramters, please refer to the paper and `quant.py` file

       
<!--
## Installation  
+ Python verseion >= 3.7.13 
+ Pytorch >= 1.12.1
+ ImageNet Dataset
+ Using docker:
```
docker run -v {local_code_loc}:{container_code_loc} -v {local_dataset_loc}:{container_dataset_loc} -it --gpus=all pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel 
```
-->
