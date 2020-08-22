#/bin/bash
CUDA_ROOT=/usr/local/cuda-9.0
TF_ROOT=/home/jd/Desktop/learn_region_grow/venv/lib/python2.7/site-packages/tensorflow
$CUDA_ROOT/bin/nvcc -DCMAKE_C_COMPILER=/usr/bin/gcc-6 tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
# g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -lcudart -L /usr/local/cuda-9.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
    -I $TF_ROOT/include \
    -I $CUDA_ROOT/include \
    -I $TF_ROOT/include/external/nsync/public \
    -lcudart \
    -L $CUDA_ROOT/lib64/ \
    -L $TF_ROOT \
    -ltensorflow_framework -O2 \
    -D_GLIBCXX_USE_CXX11_ABI=0
