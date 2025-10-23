# HAMi-core 算力切分問題追蹤與解決方案

## 問題摘要

在使用 HAMi-core 的 user-space 算力限制機制 (`CUDA_DEVICE_SM_LIMIT`) 時，觀察到以下現象：

1.  **限制效果不符預期**：當多個容器共享同一 GPU 並設定不同的算力百分比（例如 20%, 30%, 40%）時，其實際效能表現（如此處測試的 `launches/sec`）並未嚴格按照設定的比例分配，甚至可能出現 30% 和 40% 效能非常接近的情況。
2.  **100% 設定下效能異常**：當多個容器（例如 3 個）的算力均設定為 100% 時，預期它們會互相競爭導致效能下降（接近 1/3），但實際測試結果顯示每個容器都能達到接近其獨佔 GPU 時的效能。
3.  **體感時間與測量時間差異**：使用 `test_performance_benchmark` 進行測試時，透過 `cudaEventElapsedTime` 測得的 GPU 執行時間似乎比實際感受到的程式運行總時間要短。

## 分析與討論

### 1. 核心問題：算力利用率計算錯誤

* **根本原因**：原始版本的 `get_used_gpu_utilization` 函式 在計算容器算力利用率時，錯誤地**累加**了同一 GPU 上所有被 HAMi-core 監控的容器（共享同一個 `shared_region` 檔案）的 SM 利用率 (`smUtil`) 總和。
* **影響**：導致每個容器內的 `utilization_watcher` 執行緒 看到的 `userutil` 都是**總體 GPU 負載**，而非其**自身**的負載。因此，`rate_limiter` 基於錯誤的利用率和自身的 `CUDA_DEVICE_SM_LIMIT` 進行比較，做出不恰當的 `cuLaunchKernel` 延遲決策。

### 2. 解釋測試情境結果

* **情境 A (20%, 30%, 40%)**：
    * 所有容器都看到接近 90% 的**總利用率**。
    * 對於 20%、30%、40% 的容器，這個總利用率都遠超其自身限制。
    * 導致 `delta` 函數 都計算出需要大幅減少 token 分配 (`share` 趨近於 0)。
    * 所有容器都頻繁因 token 不足而被 `rate_limiter` 延遲，陷入相似的等待狀態，使得 30% 和 40% 的實際效能差異不明顯，無法按比例分配算力。
* **情境 B (100%, 100%, 100%)**：
    * `rate_limiter` 函式中有一個特殊判斷，當 `CUDA_DEVICE_SM_LIMIT` >= 100 時，會**直接返回**，不執行任何限制邏輯。
    * 因此，HAMi-core 的 user-space 限制機制失效，核心提交完全交由底層 CUDA Driver 和 GPU 硬體處理。
    * `test_performance_benchmark` 的單一實例負載可能不足以完全飽和 GPU 資源。
    * GPU 的硬體/驅動 scheduler 能夠高效地並行處理來自三個容器的請求，避免了嚴重衝突，使得每個容器都能獲得接近獨佔時的效能。

### 3. 時間測量差異

* `cudaEventElapsedTime` 主要測量 GPU 端的核心**實際執行時間**。
* 它**不包含**：
    * CPU 端的 API 呼叫開銷、迴圈開銷。
    * 由 `rate_limiter` 引入的 `nanosleep` **CPU 等待時間**。
    * `cudaEventSynchronize` 造成的 **CPU 等待 GPU 完成的時間**。
* 體感時間更接近包含以上所有因素的**牆上時鐘時間**，因此比 `cudaEventElapsedTime` 的結果長是合理的，尤其在 `rate_limiter` 啟用時差異會更明顯。

### 4. 其他潛在因素 (先前討論提及)

* **L2 快取爭用**：雖然不是導致上述測試結果的主要原因，但在高負載、多個不同類型應用混合部署時，共享 L2 快取導致的效能下降仍是軟體切分方案需要注意的固有成本。
* **記帳準確性**：SM 利用率 (`smUtil`) 本身是算力的近似指標，不同類型核心的實際算力消耗可能差異很大。

## 預期解法

* **修正 `get_used_gpu_utilization` 函式**：
    * 修改該函式，使其不再累加所有行程的 `smUtil`。
    * 應先獲取當前 `libvgpu.so` 實例對應的 `hostpid` (透過 `getpid()` 在 `shared_region` 中查找)。
    * 然後在 `nvmlDeviceGetProcessUtilization` 的結果中，**只選取** `pid` 與自身 `hostpid` 相符的行程，將其 `smUtil` 作為該容器的利用率 (`userutil`)。
    * 參考 `multiprocess_utilization_watcher.c` 中被註解掉的程式碼區塊。
* **目標**：修正後，每個容器的 `utilization_watcher` 將能準確監控**自身**的 SM 利用率，使得 `rate_limiter` 能夠基於正確的資訊和 `CUDA_DEVICE_SM_LIMIT` 進行更有效的算力限制。預期在情境 A 中，各容器的效能將更接近其設定的百分比。