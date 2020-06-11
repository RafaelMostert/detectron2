G1=4
G2=5
G3=6
G4=7
G5=8
G6=9

CUDA_VISIBLE_DEVICES=$G1 python train_lofar.py configs/lofar_detection/datasetsizes_v16_10.yaml &
CUDA_VISIBLE_DEVICES=$G2 python train_lofar.py configs/lofar_detection/datasetsizes_v16_100.yaml &
CUDA_VISIBLE_DEVICES=$G3 python train_lofar.py configs/lofar_detection/datasetsizes_v16_1000.yaml &
CUDA_VISIBLE_DEVICES=$G4 python train_lofar.py configs/lofar_detection/datasetsizes_v16_2000.yaml &
CUDA_VISIBLE_DEVICES=$G5 python train_lofar.py configs/lofar_detection/datasetsizes_v16_3000.yaml &
CUDA_VISIBLE_DEVICES=$G6 python train_lofar.py configs/lofar_detection/datasetsizes_v16_all.yaml &
