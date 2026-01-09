#!/bin/bash
# 编译 PyCANDYAlgo 模块的脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Building PyCANDYAlgo Module"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found in current directory"
    echo "Please run this script from the implementations directory"
    exit 1
fi

# 检查必要的依赖
echo ""
echo "Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 not found"
    exit 1
fi
echo "✓ Python3: $(python3 --version)"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "✗ CMake not found"
    exit 1
fi
echo "✓ CMake: $(cmake --version | head -n1)"

# Check PyTorch
if ! python3 -c "import torch" &> /dev/null; then
    echo "✗ PyTorch not found"
    echo "  Install with: pip install torch"
    exit 1
fi
echo "✓ PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# Check gflags
if ! python3 -c "import sys; import subprocess; subprocess.run(['pkg-config', '--exists', 'gflags'], check=True)" &> /dev/null; then
    echo "⚠ gflags not found (might still work if installed manually)"
else
    echo "✓ gflags found"
fi

# Check libaio
if ! ldconfig -p | grep -q libaio; then
    echo "✗ libaio not found"
    echo "  Install with: sudo apt-get install libaio-dev"
    exit 1
fi
echo "✓ libaio found"

# Check glog
if ! ldconfig -p | grep -q libglog; then
    echo "⚠ glog not found (might still work if installed manually)"
else
    echo "✓ glog found"
fi

# 创建构建目录
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# 配置 CMake
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_CXX_STANDARD=20

# 编译
echo ""
echo "Building (this may take a while)..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Using $NPROC parallel jobs"
make -j${NPROC}

# 检查生成的文件
echo ""
echo "Checking output..."
if [ -f "PyCANDYAlgo*.so" ] || [ -f "../PyCANDYAlgo*.so" ]; then
    echo "✓ PyCANDYAlgo module built successfully!"
    
    # 找到 .so 文件
    SO_FILE=$(find . -name "PyCANDYAlgo*.so" -o -name "../PyCANDYAlgo*.so" | head -n1)
    if [ -n "$SO_FILE" ]; then
        echo "  Output: $SO_FILE"
        ls -lh "$SO_FILE"
    fi
else
    echo "✗ PyCANDYAlgo.so not found!"
    exit 1
fi

# 测试导入
echo ""
echo "Testing import..."
cd ..
if python3 -c "import sys; sys.path.insert(0, '.'); import PyCANDYAlgo; print(f'✓ Import successful! Version: {PyCANDYAlgo.__version__}')"; then
    echo ""
    echo "=========================================="
    echo "Build completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run module test:"
    echo "   python3 ../test_pycandy_module.py"
    echo ""
    echo "2. The module is now available at:"
    echo "   $(pwd)/PyCANDYAlgo*.so"
else
    echo ""
    echo "⚠ Module built but import failed. Check dependencies."
    exit 1
fi
