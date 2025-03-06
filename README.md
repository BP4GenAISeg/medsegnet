




docker build -t unet-3d .

docker build --no-cache -t unet-3d .


docker run --gpus all -v $(pwd):/app -it unet-3d

