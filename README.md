run:
	docker run --gpus all -it -u $(id -u) --rm -v $(pwd):/src -w /src tensorflow/tensorflow:1.15.2-gpu-py3 python .

notebook:
	docker run --gpus all -it -u $(id -u) -p 8888:8888 -v $(pwd):/tf --rm tensorflow/tensorflow:1.15.2-gpu-py3-jupyter
