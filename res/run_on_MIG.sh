

# our server ocelot uses MIG (splits large GPU into several smaller ones virtually)

# GPU 0: A100-PCIE-40GB (UUID: GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4)                                                                      
#   MIG 3g.20gb Device 0: (UUID: MIG-GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4/1/0)                                                            
#   MIG 3g.20gb Device 1: (UUID: MIG-GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4/2/0)                                                            
# GPU 1: A100-PCIE-40GB (UUID: GPU-915536bf-10cf-6704-559e-fe0de309c9bc)                                                                      
#   MIG 3g.20gb Device 0: (UUID: MIG-GPU-915536bf-10cf-6704-559e-fe0de309c9bc/1/0)                                                            
#   MIG 3g.20gb Device 1: (UUID: MIG-GPU-915536bf-10cf-6704-559e-fe0de309c9bc/2/0)

# 1 / 2
CUDA_VISIBLE_DEVICES=MIG-GPU-c66f2b7d-c5d2-c0d1-e453-6f7f47a3b1d4/1/0 python run_sweep.py --config-name 5.7_cf10_cnn_erank_absmodel.yaml

# 3 / 4
CUDA_VISIBLE_DEVICES=MIG-GPU-915536bf-10cf-6704-559e-fe0de309c9bc/1/0 python run_sweep.py --config-name 5.7_cf10_cnn_erank_absmodel.yaml