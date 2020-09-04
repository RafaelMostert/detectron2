G1=12
G2=13
G3=14

CUDA_VISIBLE_DEVICES=$G2 python train_lofar.py \
    configs/lofar_detection/iterations_v12_100kstepLR.yaml > v12.txt &
CUDA_VISIBLE_DEVICES=$G3 python train_lofar.py \
    configs/lofar_detection/iterations_v14_100kcosineLR.yaml > v14.txt &
#CUDA_VISIBLE_DEVICES=$G1 python train_lofar.py \
#    configs/lofar_detection/iterations_v10_100kconstantLR.yaml > v10.txt &
