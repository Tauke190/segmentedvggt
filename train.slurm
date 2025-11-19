#!/bin/bash
#SBATCH --job-name=timm_resnet
#SBATCH --gres=gpu:1
    #SBATCH --constraint=gmem48
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --output=timm_resnet%j.out
#SBATCH --error=timm_resnet%j.err
#SBATCH --time=24:00:00


module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"

conda activate vggt311

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

cd /home/av354855/VLMtoresnet

python -c "import torch; print('Torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -u seg_head_training_coco --train_path /home/c3-0/datasets/coco/train2017 --annotation_path /home/c3-0/datasets/coco/annotations/instances_train2017.json --val_path /home/c3-0/datasets/coco/val2017 --val_annotation_path /home/c3-0/datasets/coco/annotations/instances_val2017.json --model_name resnet50 --batch_size 16 --num_epochs 50 --learning_rate 0.001 --output_dir ./timm_resnet_output
echo "Job finished at $(date)"