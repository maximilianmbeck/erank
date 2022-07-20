

# our server ocelot uses MIG (splits large GPU into several smaller ones virtually)
### OCELOT
# GPU 0: A100-PCIE-40GB (UUID: GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4)                                                                      
#   MIG 3g.20gb Device 0: (UUID: MIG-GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4/1/0)                                                            
#   MIG 3g.20gb Device 1: (UUID: MIG-GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4/2/0)                                                            
# GPU 1: A100-PCIE-40GB (UUID: GPU-915536bf-10cf-6704-559e-fe0de309c9bc)                                                                      
#   MIG 3g.20gb Device 0: (UUID: MIG-GPU-915536bf-10cf-6704-559e-fe0de309c9bc/1/0)                                                            
#   MIG 3g.20gb Device 1: (UUID: MIG-GPU-915536bf-10cf-6704-559e-fe0de309c9bc/2/0)

# 1 / 2
CUDA_VISIBLE_DEVICES=MIG-GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4/1/0 python run_sweep.py --config-name 5.7_cf10_cnn_erank_absmodel.yaml

# 3
CUDA_VISIBLE_DEVICES=MIG-GPU-915536bf-10cf-6704-559e-fe0de309c9bc/1/0 python run_sweep.py --config-name 5.17_cf10_cnn_erank_absmodel_datasetsize.yaml
# 4
CUDA_VISIBLE_DEVICES=MIG-GPU-915536bf-10cf-6704-559e-fe0de309c9bc/2/0 python run_sweep.py --config-name 5.18_cf10_fc_erank_absmodel_datasetsize.yaml

### BADGER
# GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-624f9f56-c3ca-cd9d-4faf-485756bca59f)
#   MIG 3g.20gb     Device  0: (UUID: MIG-f5701ea0-d53d-50fd-a077-180824228d53)
#   MIG 3g.20gb     Device  1: (UUID: MIG-835ae22c-b7c9-5474-b65c-c3b7e2057b10)
# GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-5c5be6e6-71fd-b0a7-6da8-957e33a615bd)
#   MIG 3g.20gb     Device  0: (UUID: MIG-0c82aa27-fb81-54f0-89ed-a6d4e6f6dc24)
#   MIG 3g.20gb     Device  1: (UUID: MIG-65fd9fa9-8535-58e9-a67d-dbaae24a0a57)

# 0
CUDA_VISIBLE_DEVICES=MIG-f5701ea0-d53d-50fd-a077-180824228d53 python run_sweep.py --config-name 5.11_cf10_cnn_erank_absmodel_sgd.yaml
# 1
CUDA_VISIBLE_DEVICES=MIG-835ae22c-b7c9-5474-b65c-c3b7e2057b10 python run_sweep.py --config-name 5.11_cf10_cnn_erank_absmodel_sgd.yaml
# 2
CUDA_VISIBLE_DEVICES=MIG-0c82aa27-fb81-54f0-89ed-a6d4e6f6dc24 python run_sweep.py --config-name 5.10_fmnist_fc_erank_absmodel_sgd.yaml
# 3
CUDA_VISIBLE_DEVICES=MIG-65fd9fa9-8535-58e9-a67d-dbaae24a0a57 python run_sweep.py --config-name 5.12_cf10_fc_erank_absmodel_sgd.yaml