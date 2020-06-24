G1=10
G2=11
G3=12
G4=13
G5=14
G6=15

CUDA_VISIBLE_DEVICES=$G1 python train_lofar.py configs/lofar_detection/datasetsizes_v17_10.yaml &
CUDA_VISIBLE_DEVICES=$G2 python train_lofar.py configs/lofar_detection/datasetsizes_v17_100.yaml &
CUDA_VISIBLE_DEVICES=$G3 python train_lofar.py configs/lofar_detection/datasetsizes_v17_1000.yaml &
CUDA_VISIBLE_DEVICES=$G4 python train_lofar.py configs/lofar_detection/datasetsizes_v17_2000.yaml &
CUDA_VISIBLE_DEVICES=$G5 python train_lofar.py configs/lofar_detection/datasetsizes_v17_3000.yaml &
CUDA_VISIBLE_DEVICES=$G6 python train_lofar.py configs/lofar_detection/datasetsizes_v17_all.yaml &
