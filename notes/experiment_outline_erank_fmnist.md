# Experiment Outline erank FashionMNIST

## Sanity checks

- Sanity check run same run twice: successful
- Sanity check compute gradients through erank: got CUDA error
    - failed for        pytorch                   1.10.0          py3.9_cuda11.3_cudnn8.2.0_0
    - succesful for     pytorch                   1.7.1           py3.8_cuda11.0.221_cudnn8.0.5_0
    - try to install pytorch 
    - successfull with pytorch 1.8.2 (LTS)
### 30.06.

Experiments run:
1. abs model parameters
2. pretraindiff normalize to 1 
3. hypsearch optimizer, weight_decay, lr, batch_size

Planned: 
4. pretraindiff rescale updatestep to average length of over vectors