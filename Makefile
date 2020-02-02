run:
	docker run --gpus all -it --rm -v $(pwd)/src:/tmp -w /tmp tensorflow/tensorflow:latest-gpu-py3 python ./script.py

notebook:
	docker run --gpus all -it  -p 8888:8888 --rm tensorflow/tensorflow:latest-gpu-py3-jupyter