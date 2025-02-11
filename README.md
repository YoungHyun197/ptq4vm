# [WACV 2025] PTQ4VM: Post-training Quantization for Visual Mamba

This is official code for the paper [PTQ4VM](https://arxiv.org/abs/2412.20386).

PTQ4VM can be applied to various Visual Mamba backbones, converting the pretrained model to a quantized format in under 15 minutes without notable quality degradation.


## Install
1. Setting conda
```
conda create -n ptq4vm python=3.10 -y
conda activate ptq4vm
```

2. Clone the PTQ4VM repository
```
git clone https://github.com/YoungHyun197/ptq4vm
cd ptq4vm
```

3. Install the dependencies
```
pip install -r requirements.txt
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.2.0.post1
```

4. Replace core implementation of Mamba
```
cp -rf mamba-1p1p1/mamba_ssm /opt/conda/lib/python3.10/site-packages
```

5. Install the CUDA kernel
```
python ./cuda_measure/setup_vim_GEMM.py install
```

## How to use PTQ4VM
Here we use Vision Mamba (Vim) model as an example. Before applying ptq4vm, prepare a pre-trained model. You can download the ckpt file from this [url](https://huggingface.co/hustvl/Vim-tiny-midclstok).

### Generate activation smoothing scale  
```
torchrun --nproc_per_node 1 generate_act_scale.py --resume [model-path] --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path [imagenet path] --batch-size 256
```

### Joint Learning of Smoothing Scale and Step size (JLSS)
```
torchrun --nproc_per_node 1 quant.py --eval --resume [model-path] --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path [imagenet-path] --act_scales [smoothing-path] --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 16 --n-lvw 16 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2
```
For experimental details and hyper-paramters, please refer to the paper and `quant.py` file



### Speedup using CUDA kernel
1. Check the layer-wise acceleration
```
python cuda_sandbox.py
```

2. Check the end-to-end acceleration
```
torchrun --nproc_per_node 1 quant.py --eval --time_compare --resume [model-path] --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path [imagenet-path] --act_scales [smoothing-path] --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 16 --n-lvw 16 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2
```

## Reference
[Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)

This example code is based on [Vim](https://github.com/hustvl/Vim).
## Cite
If you find our code or PTQ4VM paper useful for your research, please consider citing:
```
@article{cho2024ptq4vm,
  title={PTQ4VM: Post-Training Quantization for Visual Mamba},
  author={Cho, Younghyun and Lee, Changhun and Kim, Seonggon and Park, Eunhyeok},
  journal={arXiv preprint arXiv:2412.20386},
  year={2024}
}
```
       
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
