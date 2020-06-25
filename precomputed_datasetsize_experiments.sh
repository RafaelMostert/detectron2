G1=10
G2=11
G3=12
G4=13
G5=14
G6=15

CUDA_VISIBLE_DEVICES=$G1 python train_lofar.py configs/lofar_detection/precomputed_datasetsizes_v18_10.yaml &
CUDA_VISIBLE_DEVICES=$G2 python train_lofar.py configs/lofar_detection/precomputed_datasetsizes_v18_100.yaml &
CUDA_VISIBLE_DEVICES=$G3 python train_lofar.py configs/lofar_detection/precomputed_datasetsizes_v18_1000.yaml &
CUDA_VISIBLE_DEVICES=$G4 python train_lofar.py configs/lofar_detection/precomputed_datasetsizes_v18_2000.yaml &
CUDA_VISIBLE_DEVICES=$G5 python train_lofar.py configs/lofar_detection/precomputed_datasetsizes_v18_3000.yaml &
CUDA_VISIBLE_DEVICES=$G6 python train_lofar.py configs/lofar_detection/precomputed_datasetsizes_v18_all.yaml &
