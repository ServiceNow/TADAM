# TADAM

## Set up docker
go to folder docker in this project, execute
docker build -f Dockerfile -t boris_tadam .

launch docker

NV_GPU=0 nvidia-docker run -p 1250:8888 -p 1251:6006 -p 1252:6007 -p 1253:6008 -v /mnt/datasets/public/:/mnt/datasets/public/ -v /mnt/home/boris:/mnt/home/boris -t -d --name boris_tadam_explore boris_tadam

iPython session should be available at http://machine_ip:1240/, password is "default". Datasets are mapped inside docker in /mnt/datasets/public/ folder.
