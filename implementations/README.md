# Algorithm Implementations - C++ 源码和编译

这个目录**仅包含 C++ 源码和编译配置**，用于生成 **PyCANDYAlgo.so** 扩展模块。

**职责划分**:

- 本目录 (`implementations/`)：C++ 源码、编译脚本、第三方库源码
- `benchmark_anns/bench/algorithms/`：Python wrapper 代码，调用编译好的 `.so` 文件

**特点**:

- 独立编译项目，包含所有第三方库源码
- 构建目标：生成 `PyCANDYAlgo.so` Python 扩展模块
- **仅支持 CPU**，不需要 CUDA
- Python wrapper 代码位于 `bench/algorithms/`

## 🚀 快速开始

### 方式 1: 一键构建所有算法（推荐）

```bash
# 1. 安装系统依赖
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake libgflags-dev libboost-all-dev libomp-dev

# macOS
brew install cmake gflags boost libomp

# 2. 安装 Python 依赖
pip install torch numpy pybind11

# 3. 构建所有算法
./build_all.sh

# 4. 安装 Python 包
./install_packages.sh
```

**构建选项**:

```bash
./build_all.sh --install              # 构建并自动安装
./build_all.sh --skip-pycandy         # 跳过 PyCANDY
./build_all.sh --skip-third-party     # 跳过第三方库
./build_all.sh --skip-vsag            # 跳过 VSAG
./build_all.sh --help                 # 显示帮助
```

### 方式 2: 分别构建各个算法

#### 构建 PyCANDY 算法

```bash
./build.sh
```

生成 `PyCANDYAlgo.cpython-310-x86_64-linux-gnu.so`

#### 构建第三方库 (GTI, IP-DiskANN, PLSH)

```bash
# GTI
cd gti/GTI
mkdir -p build && cd build
cmake .. && make -j$(nproc) && make install

# IP-DiskANN
cd ipdiskann
mkdir -p build && cd build
cmake .. && make -j$(nproc) && make install

# PLSH
cd plsh
mkdir -p build && cd build
cmake .. && make -j$(nproc) && make install
```

#### 构建 VSAG

```bash
cd vsag
make release                          # 构建 release 版本
make pyvsag PY_VERSION=3.10          # 构建 Python wheel
pip install wheelhouse/pyvsag*.whl   # 安装
```

### 验证安装

```bash
python3 -c "import PyCANDYAlgo; print('✅ PyCANDYAlgo OK')"
python3 -c "import pyvsag; print('✅ pyvsag OK')"
```

**故障排除**: 如遇到 `ImportError: undefined symbol` 错误，删除旧版本后重新安装：

```bash
rm -f ~/.local/lib/python3.10/site-packages/PyCANDYAlgo*.so
cp PyCANDYAlgo*.so $(python3 -c "import site; print(site.USER_SITE)")/
```

## 目录结构

```
implementations/                   # C++ 源码和编译配置
├── bindings/PyCANDY.cpp          # pybind11 绑定实现
├── candy/                         # CANDY 算法 C++ 源码
├── faiss/                         # Faiss 源码
├── diskann-ms/                    # DiskANN 源码
├── puck/                          # Puck 源码
├── SPTAG/                         # SPTAG 源码
├── gti/                           # GTI 源码
├── ipdiskann/                     # IP-DiskANN 源码
├── plsh/                          # PLSH 源码
├── vsag/                          # VSAG 源码
├── pybind11/                      # pybind11 库
├── build.sh                       # PyCANDY 构建脚本
├── build_all.sh                   # 一键构建所有算法脚本
├── install_packages.sh            # 安装 Python 包脚本
├── CMakeLists.txt                 # CMake 配置
├── setup.py                       # PyCANDYAlgo 打包配置
└── README.md                      # 本文件
```

**Python wrapper 层**在 `bench/algorithms/` 目录，提供友好的 NumPy 接口。

## 第三方库管理

本目录包含以下第三方库源码：

| 库             | 说明                | 构建方式               | Python 包     |
| -------------- | ------------------- | ---------------------- | ------------- |
| **GTI**        | 基于图的树索引      | CMake                  | 无 (C++ only) |
| **IP-DiskANN** | 插入优先的 DiskANN  | CMake                  | 无 (C++ only) |
| **PLSH**       | 并行局部敏感哈希    | CMake                  | 无 (C++ only) |
| **VSAG**       | 向量搜索加速引擎    | Makefile + wheel       | pyvsag        |
| **Faiss**      | Meta 向量相似度搜索 | CMake (集成到 PyCANDY) | -             |
| **DiskANN**    | 微软磁盘索引        | CMake (集成到 PyCANDY) | -             |
| **SPTAG**      | 微软空间分区树和图  | CMake (集成到 PyCANDY) | -             |
| **Puck**       | 百度向量搜索引擎    | CMake (集成到 PyCANDY) | -             |

**构建分类**：

1. **通过 PyCANDY 构建**: Faiss, DiskANN, SPTAG, Puck → 生成 `PyCANDYAlgo.so`
2. **独立 CMake 构建**: GTI, IP-DiskANN, PLSH → 生成 C++ 库
3. **独立 Makefile + wheel**: VSAG → 生成 `pyvsag-*.whl`

## 手动构建（可选）

如果 `build.sh` 不适合你的环境：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF
make -j$(nproc)
cd ..
```

**构建流程**: 脚本会按顺序构建 GTI（含 n2）→ IP-DiskANN → PLSH → PyCANDYAlgo 主模块

## PyCANDYAlgo 模块内容

编译后的 `PyCANDYAlgo.so` 包含：

### 算法实现

- **CANDY 算法**: HNSWNaive, LSH, LSHAPG, FlatIndex, NNDescent, SPTAGIndex, DPGIndex,
  CongestionDropIndex
- **Faiss**: 索引工厂，HNSW, IVFPQ, LSH 等
- **DiskANN**: 动态内存索引，支持高效插入和搜索
- **Puck**: 百度搜索引擎接口
- **GTI**: 基于图的树索引，支持对数级更新
- **IP-DiskANN**: 插入优先的 DiskANN 变体
- **PLSH**: 并行局部敏感哈希

### 核心接口

1. **ConfigMap**: 配置管理，支持 int64/double/string，与 Python dict 互转
1. **AbstractIndex**: 统一索引接口，支持插入、删除、查询、维护
1. **并发队列**: NumpyIdxPair, NumpyIdxQueue, IdxQueue（SPSC队列）
1. **Faiss/DiskANN/Puck**: 直接封装对应库的 C++ API

## 使用方式

### 方式 1: 底层 C++ 接口（直接使用 PyCANDYAlgo）

```python
import PyCANDYAlgo

index = PyCANDYAlgo.createIndex("HNSWNaive", dim=128)
config = PyCANDYAlgo.newConfigMap()
config.edit("vecDim", 128)
config.edit("M", 16)

db = PyCANDYAlgo.loadTensorFromFile("data.fvecs")
index.loadInitialTensor(db, config)

query = PyCANDYAlgo.loadTensorFromFile("query.fvecs")
results = index.searchTensor(query, 10)
```

### 方式 2: Python Wrapper（推荐）

```python
from benchmark_anns.bench.algorithms import CANDYWrapper

wrapper = CANDYWrapper(index_type="HNSWNaive", metric="euclidean")
wrapper.setup(dtype="float32", max_pts=100000, ndims=128)
wrapper.insert(vectors, ids)
I, D = wrapper.query(queries, k=10)
```

### 方式 3: 在 benchmark 中使用

```python
from benchmark_anns.bench import get_algorithm

algo = get_algorithm("candy_flat")  # 自动从 registry 加载
algo = get_algorithm("faiss_hnsw")
algo = get_algorithm("diskann")
```

**注意**: Python wrapper（`bench/algorithms/`）提供更友好的 NumPy 接口和错误处理。

## 故障排除

### 常见问题

**ImportError: PyCANDYAlgo**

```bash
ls -la PyCANDYAlgo*.so  # 检查是否编译成功
python3 -c "import sys; print(sys.path)"  # 检查路径
```

**符号未定义错误**

- **原因**: `candy/Utils/` 目录缺少 .cpp 实现文件
- **解决**: 从主项目 `src/Utils/` 复制对应的 .cpp 文件到 `candy/Utils/`
- 需要的文件: `IntelliLog.cpp`, `IntelliTimeStampGenerator.cpp`, `MemTracker.cpp`, `UtilityFunctions.cpp`
  及 `Meters/` 下的实现文件

**CMake 找不到 Torch**

```bash
pip install torch
export CMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
```

**编译时内存不足**

```bash
make -j2  # 减少并行数，而非 make -j$(nproc)
```
