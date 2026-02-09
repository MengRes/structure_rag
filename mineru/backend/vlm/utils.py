import os

from loguru import logger
from packaging import version

from mineru.utils.check_sys_env import is_windows_environment, is_linux_environment
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram


def enable_custom_logits_processors() -> bool:
    import torch
    from vllm import __version__ as vllm_version

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # 正确计算Compute Capability
        compute_capability = f"{major}.{minor}"
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        compute_capability = "8.0"
    else:
        logger.info("CUDA not available, disabling custom_logits_processors")
        return False

    # 安全地处理环境变量
    vllm_use_v1_str = os.getenv('VLLM_USE_V1', "1")
    if vllm_use_v1_str.isdigit():
        vllm_use_v1 = int(vllm_use_v1_str)
    else:
        vllm_use_v1 = 1

    if vllm_use_v1 == 0:
        logger.info("VLLM_USE_V1 is set to 0, disabling custom_logits_processors")
        return False
    elif version.parse(vllm_version) < version.parse("0.10.1"):
        logger.info(f"vllm version: {vllm_version} < 0.10.1, disable custom_logits_processors")
        return False
    elif version.parse(compute_capability) < version.parse("8.0"):
        if version.parse(vllm_version) >= version.parse("0.10.2"):
            logger.info(f"compute_capability: {compute_capability} < 8.0, but vllm version: {vllm_version} >= 0.10.2, enable custom_logits_processors")
            return True
        else:
            logger.info(f"compute_capability: {compute_capability} < 8.0 and vllm version: {vllm_version} < 0.10.2, disable custom_logits_processors")
            return False
    else:
        logger.info(f"compute_capability: {compute_capability} >= 8.0 and vllm version: {vllm_version} >= 0.10.1, enable custom_logits_processors")
        return True


def set_lmdeploy_backend(device_type: str) -> str:
    if device_type.lower() in ["ascend", "maca", "camb"]:
        lmdeploy_backend = "pytorch"
    elif device_type.lower() in ["cuda"]:
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        if is_windows_environment():
            lmdeploy_backend = "turbomind"
        elif is_linux_environment():
            major, minor = torch.cuda.get_device_capability()
            compute_capability = f"{major}.{minor}"
            if version.parse(compute_capability) >= version.parse("8.0"):
                lmdeploy_backend = "pytorch"
            else:
                lmdeploy_backend = "turbomind"
        else:
            raise ValueError("Unsupported operating system.")
    else:
        raise ValueError(f"Unsupported lmdeploy device type: {device_type}")
    return lmdeploy_backend


def set_default_gpu_memory_utilization() -> float:
    import torch
    env_util = os.getenv("MINERU_GPU_MEMORY_UTILIZATION")
    if env_util is not None:
        try:
            val = float(env_util)
            if 0 < val <= 1.0:
                logger.info(f"Using MINERU_GPU_MEMORY_UTILIZATION={val}")
                return val
        except ValueError:
            pass
    from vllm import __version__ as vllm_version
    device = get_device()
    gpu_memory = get_vram(device)
    desired = 0.5
    if version.parse(vllm_version) >= version.parse("0.11.0") and gpu_memory <= 8:
        desired = 0.7
    # 若当前空闲显存不足，则按空闲显存比例上限压低 utilization，避免 vLLM 启动报错
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            if hasattr(torch.cuda, "mem_get_info"):
                free_gb = torch.cuda.mem_get_info(device)[0] / (1024 ** 3)
                total_gb = torch.cuda.mem_get_info(device)[1] / (1024 ** 3)
                if total_gb > 0 and free_gb < desired * total_gb:
                    cap = 0.9 * free_gb / total_gb
                    util = min(desired, max(0.1, round(cap, 2)))
                    logger.info(f"GPU free {free_gb:.2f}/{total_gb:.2f} GiB, using gpu_memory_utilization={util} (desired {desired})")
                    return util
        except Exception as e:
            logger.debug(f"Could not get GPU free memory: {e}")
    return desired


def set_default_batch_size() -> int:
    try:
        device = get_device()
        gpu_memory = get_vram(device)

        if gpu_memory >= 16:
            batch_size = 8
        elif gpu_memory >= 8:
            batch_size = 4
        else:
            batch_size = 1
        logger.info(f'gpu_memory: {gpu_memory} GB, batch_size: {batch_size}')

    except Exception as e:
        logger.warning(f'Error determining VRAM: {e}, using default batch_ratio: 1')
        batch_size = 1
    return batch_size