CKPT_DIR="/home/ivan/projects/TAMI-template/logs/DIV2K/SRNet/2024-10-28/hello_pytorch/checkpoints/best.cpkt"
IMAGE_DIR="/home/ivan/projects/TAMI-template/logs/DIV2K/SRNet/2024-10-28/hello_pytorch"
python src/test.py --ckpt_path ${CKPT_DIR} --output_path ${IMAGE_DIR} --sampling 2 --dataset_path "/home/ivan/datasets/DIV2K"
