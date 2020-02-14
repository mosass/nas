run:
	docker run --gpus all -it -u $(id -u) --rm -v $(pwd):/src -w /src tensorflow/tensorflow:latest-gpu-py3 python ./script.py

notebook:
	docker run --gpus all -it -u $(id -u) -p 8888:8888 -v $(pwd):/tf --rm tensorflow/tensorflow:latest-gpu-py3-jupyter
