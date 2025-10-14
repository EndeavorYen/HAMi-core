# HAMi-core —— CUDA 环境的 Hook 库

[English](README.md) | 中文

## 介绍

HAMi-core 是一个容器内的 GPU 资源控制器，已被 [HAMi](https://github.com/Project-HAMi/HAMi) 和 [volcano](https://github.com/volcano-sh/devices) 等项目采用。

<img src="./docs/images/hami-arch.png" width = "600" /> 

## 特性

HAMi-core 具有以下特性：
1. GPU 显存虚拟化
2. 通过自实现的时间片方式限制设备利用率
3. 实时设备利用率监控

![image](docs/images/sample_nvidia-smi.png)

## 设计原理

HAMi-core 通过劫持 CUDA-Runtime(libcudart.so) 和 CUDA-Driver(libcuda.so) 之间的 API 调用来实现功能，如下图所示：

<img src="./docs/images/hami-core-position.png" width = "400" />

## 在Docker中编译

```bash
make build-in-docker
```

## 使用方法

_CUDA_DEVICE_MEMORY_LIMIT_ 用于指定设备内存的上限（例如：1g、1024m、1048576k、1073741824）

_CUDA_DEVICE_SM_LIMIT_ 用于指定每个设备的 SM 利用率百分比

```bash
# 为挂载的设备添加 1GiB 内存限制并将最大 SM 利用率设置为 50%
export LD_PRELOAD=./libvgpu.so
export CUDA_DEVICE_MEMORY_LIMIT=1g
export CUDA_DEVICE_SM_LIMIT=50
```

## Docker镜像使用

```bash
# 构建 Docker 镜像
docker build . -f=dockerfiles/Dockerfile -t cuda_vmem:tf1.8-cu90

# 配置容器的 GPU 设备和库挂载选项
export DEVICE_MOUNTS="--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidiactl:/dev/nvidiactl"
export LIBRARY_MOUNTS="-v /usr/cuda_files:/usr/cuda_files -v $(which nvidia-smi):/bin/nvidia-smi"

# 运行容器并查看 nvidia-smi 输出
docker run ${LIBRARY_MOUNTS} ${DEVICE_MOUNTS} -it \
    -e CUDA_DEVICE_MEMORY_LIMIT=2g \
    -e LD_PRELOAD=/libvgpu/build/libvgpu.so \
    cuda_vmem:tf1.8-cu90 \
    nvidia-smi
```

运行后，您将看到类似以下的 nvidia-smi 输出，显示内存被限制在 2GiB：

```
...
[HAMI-core Msg(1:140235494377280:libvgpu.c:836)]: Initializing.....
Mon Dec  2 04:38:12 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.107.02             Driver Version: 550.107.02     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        Off |   00000000:03:00.0 Off |                  N/A |
| 30%   36C    P8              7W /  170W |       0MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
[HAMI-core Msg(1:140235494377280:multiprocess_memory_limit.c:497)]: Calling exit handler 1
```

## 日志级别

使用环境变量 LIBCUDA_LOG_LEVEL 来设置日志的可见性

| LIBCUDA_LOG_LEVEL | 描述 |
| ----------------- | ----------- |
|  0          | 仅错误信息 |
|  1(默认),2   | 错误、警告和消息 |
|  3          | 信息、错误、警告和消息 |
|  4          | 调试、错误、警告和消息 |

## 测试原始API

```bash
./test/test_alloc
```


-----

## GPU Slicing 測試

### 動態調整與功能驗證 (Runtime Verification and Dynamic Adjustment)

本章節將引導您如何在容器 (Container) 環境中，驗證 Hami-core 的核心功能：

1.  **顯存限制 (VRAM Limit)**：驗證容器內的 GPU 顯存是否被成功限制在指定的大小。
2.  **動態算力調整 (Dynamic Compute Limit)**：驗證如何在容器執行期間，動態地調整 GPU 的算力限制。

#### 前置準備：編譯專案

在進行任何測試之前，請先確保您已經在專案的根目錄下執行過編譯指令。這個指令會利用 Docker 建立一個乾淨的編譯環境，並將編譯產物 (包含 `libvgpu.so` 和所有測試工具) 放置在主機的 `build/` 目錄下。

```bash
make build-in-docker
```

#### 測試 1：驗證 VRAM 限制

此測試將啟動一個 VRAM 上限為 1GB 的容器，並執行一個壓力測試程式，該程式會不斷申請記憶體直到觸發「記憶體不足」(Out of Memory) 錯誤為止。

1.  **啟動容器**
    使用以下指令啟動容器。我們將主機上的 `build` 目錄掛載到容器的 `/test_build` 路徑，以便執行最新的測試程式。

    ```bash
    docker run --rm -it \
        --gpus all \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v $(pwd)/build:/test_build \
        -e CUDA_DEVICE_MEMORY_LIMIT=1g \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        cuda_vmem:tf1.8-cu90 \
        bash
    ```

2.  **執行測試程式**
    在容器內，執行 `test_vram_limit` 程式。

    ```bash
    # 在 Container 內執行
    /test_build/test_vram_limit
    ```

3.  **預期結果**
    程式會開始持續分配記憶體，並在總分配量接近 1024MB 時，成功捕獲 `CUDA_ERROR_OUT_OF_MEMORY` 錯誤並正常退出。這證明 VRAM 限制功能已成功生效。

    ```
    開始 VRAM 分配測試，每次增加 100MB...
    ...
    第 10 次分配: 成功分配 100MB. 目前總分配量: 1000.00 MB

    =========================================================
    成功捕獲預期的 CUDA_ERROR_OUT_OF_MEMORY 錯誤！
    VRAM 限制功能正常運作。
    在 OOM 之前，總共成功分配了: 1000.00 MB
    =========================================================

    VRAM 限制測試完成。
    ```

#### 測試 2：驗證動態算力調整

此測試將演示如何在容器執行期間，從外部動態調整 GPU 的算力限制。

1.  **啟動容器與壓力測試 (終端機 A)**
    啟動一個容器，設定**初始算力上限為 80%**，並在其中執行一個無限迴圈的 GPU 壓力測試程式。

    ```bash
    # 啟動一個名為 hami-test 的容器
    docker run --rm -it --name hami-test \
        --gpus all \
        --mount type=tmpfs,destination=/tmp/vgpulock \
        -v $(pwd)/build:/test_build \
        -e CUDA_DEVICE_SM_LIMIT=80 \
        -e LD_PRELOAD=/test_build/libvgpu.so \
        cuda_vmem:tf1.8-cu90 \
        bash

    # 進入容器後，立即啟動壓力測試
    # 這個程式會持續運行
    /test_build/test_runtime_launch
    ```

2.  **在主機上監控 GPU**
    在您的**主機 (Host)** 上打開一個新的終端機，執行以下指令來監控 GPU 狀態。你會看到 GPU 利用率 (`GPU-Util`) 穩定在 80% 左右。

    ```bash
    watch -n 1 nvidia-smi
    ```

3.  **進入容器並修改算力 (終端機 B)**
    再打開一個新的終端機，使用 `docker exec` 進入**正在運行的** `hami-test` 容器。

    ```bash
    docker exec -it hami-test bash
    ```

    進入容器後，使用我們擴充過的 `shrreg-tool` 工具來動態修改算力限制。例如，將 0 號 GPU 的算力限制從 80% **降至 20%**。

    ```bash
    # 在終端機 B (容器內) 執行
    /test_build/shrreg-tool --set-sm-limit 0 20
    ```

4.  **觀察結果**
    此時，切換回你的 `nvidia-smi` 監控視窗。你會看到 GPU 利用率在幾秒鐘內從 80% 明顯下降，並穩定在 20% 附近。

5.  **再次調整**
    你可以在終端機 B 中再次執行指令，將算力調高，例如調至 70%。

    ```bash
    # 在終端機 B (容器內) 執行
    /test_build/shrreg-tool --set-sm-limit 0 70
    ```

    GPU 利用率也會隨之回升至 70% 左右。這個流程完整地驗證了算力限制的動態調整能力。