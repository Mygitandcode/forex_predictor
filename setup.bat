@echo off
REM ============================================================
REM setup.bat — Windows 环境一键初始化脚本
REM ============================================================
REM 使用方法: 双击运行 或 在 PowerShell 中执行 .\setup.bat
REM ============================================================

echo ========================================
echo   MT4 Forex Predictor - 环境初始化
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.9+
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [✓] Python 已安装

REM 创建虚拟环境
if not exist "venv" (
    echo [*] 创建虚拟环境...
    python -m venv venv
    echo [✓] 虚拟环境已创建
) else (
    echo [✓] 虚拟环境已存在
)

REM 激活并安装依赖
echo [*] 安装 Python 核心依赖（16个包）...
call venv\Scripts\activate.bat
echo.
echo Python 版本:
python --version
echo.
pip install --upgrade pip -q
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 核心依赖安装失败，请检查网络或 requirements.txt
    pause
    exit /b 1
)
echo [✓] 16 个核心依赖安装完成
echo.

REM 单独尝试安装 TA-Lib（大概率失败，不阻塞）
echo [*] 尝试安装 TA-Lib...
pip install TA-Lib>=0.4.28 2>&1
if %errorlevel% equ 0 (
    echo [✓] TA-Lib 安装成功
) else (
    echo [!] TA-Lib pip 安装失败（Windows 缺少 C 库，这是正常的）
    echo.
    echo   手动安装 TA-Lib 方法:
    echo   1. 打开: https://github.com/cgohlke/talib-build/releases
    echo   2. 下载: TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
    echo   3. 在 venv 激活状态下执行:
    echo      pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
    echo.
    echo   暂时装不上也没关系，FeatureBuilder 会自动跳过 TA-Lib 特征
)

REM 创建必要目录
if not exist "models" mkdir models
if not exist "data_store" mkdir data_store
if not exist "mlruns" mkdir mlruns
if not exist "autogluon_models" mkdir autogluon_models
if not exist "shap_output" mkdir shap_output
if not exist "cluster_output" mkdir cluster_output
echo [✓] 目录结构已创建

REM ta-lib 提示（如果前面安装成功就跳过）
echo.
echo ========================================
echo   环境初始化完成!
echo ========================================
echo.
echo  后续步骤:
echo   1. 激活环境: venv\Scripts\activate
echo   2. 启动 MLflow: python -m serving.mlflow_tracking serve
echo   3. 导入数据: python -m features.feature_pipeline --data-dir ^<MT4数据目录^>
echo   4. 开始训练: python -m training.stage1_autogluon --symbol EURUSD
echo   5. 启动服务: uvicorn serving.fastapi_app:app --reload
echo.
pause
