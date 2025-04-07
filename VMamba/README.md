
## Install
To properly run our code, you need to create a new virtual environment because VMamba uses a different selective_scan kernel than Vim.

1. Setting conda
```
conda create -n ptq4vm_vmamba python=3.10 -y
conda activate ptq4vm_vmamba
```

2. Clone the PTQ4VM repository
  
    If you have already cloned, you can skip this step.
```
git clone https://github.com/YoungHyun197/ptq4vm
cd ptq4vm/VMamba
```

3. Install the dependencies
```
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
pip install --pre -U triton
```

## How to use PTQ4VM
Before applying ptq4vm, prepare a pre-trained model. You can download the model from this [url](https://github.com/MzeroMiko/VMamba).

```
cd classification
```

### Generate activation smoothing scale  
```
torchrun --nproc_per_node 1 generate_act_scale.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path [imagenet path] --output /tmp --pretrained [model-path] --batch-size 256 --scales-output-path ./act_scales
```

### Joint Learning of Smoothing Scale and Step size (JLSS)
```
torchrun --nproc_per_node 1 quant.py --cfg configs/vssm/vmambav2v_tiny_224.yaml  --data-path [imagenet-path] --output /tmp --pretrained [model-path] --act_scales [smoothing-path] --batch-size 256 --qmode ptq4vm  --train-batch 32 --epochs 50 --lr-a 1e-4 --lr-w 1e-4 --lr-s 1e-4 --n-lva 16 --n-lvw 16 --alpha 0.65
```
- n-lva (n-lvw) : activation (weight) quantizaiton levels (8/6/4-bit: 256/64/16)  
  - Refer to the `initialize()` function of Q_Linear and Q_Act classes in ptq4vm/quantizer.py
- lr-a (lr-w, lr-s) : learning rates of activation (weight, smooth scale) step size

For experimental details and hyper-paramters, please refer to the paper and `quant.py` file


## Reference
[VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)

This example code is based on [VMamba](https://github.com/MzeroMiko/VMamba).

