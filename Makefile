# 1024 4096 8192 16384 32768 131072
SEED = 0
POINTS_NUM = 1024
CLUSTERS_NUM = 10
MAX_ITER = 1000

INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_61 --ptxas-options=-v --use_fast_math

all: kmeans_cpp kmeans_cu generate_data run_kmeans_cpp run_kmeans_cu

kmeans_cu: kmeans.cu 
	nvcc kmeans.cu -o kmeans_cu $(INC) $(NVCCFLAGS) $(LIB)

run_kmeans_cu: kmeans_cu
	./kmeans_cu $(POINTS_NUM) $(CLUSTERS_NUM) $(MAX_ITER)

run_kmeans_cpp: kmeans.o utils.o
	./kmeans_cpp $(POINTS_NUM) $(CLUSTERS_NUM) $(MAX_ITER)

kmeans_cpp: kmeans.o utils.o
	g++ kmeans.o utils.o -o kmeans_cpp

kmeans_cpp_debug: kmeans.o utils.o
	g++ -g kmeans.o utils.o -o kmeans_cpp_debug

kmeans.o: kmeans.cpp utils.h
	g++ -c kmeans.cpp

utils.o: utils.cpp utils.h
	g++ -c utils.cpp

venv/bin/activate: requirements.txt
	python3.10 -m venv --upgrade-deps venv
	./venv/bin/pip install -r requirements.txt

generate_data: venv/bin/activate
	./venv/bin/python3 generate-data.py $(SEED) $(POINTS_NUM)

plot_data: venv/bin/activate
	./venv/bin/python3 plot-data.py $(CLUSTERS_NUM)

clean:
	rm -f kmeans_cpp kmeans_cu utils.o kmeans.o kmeans_cpp_debug
	rm -rf __pycache__
	rm -rf ./venv
