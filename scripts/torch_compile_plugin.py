# -*- coding: utf-8 -*-
import torch
import sys
import time
import os
import traceback
import logging
import hashlib
import gradio as gr
from types import MethodType
from modules import scripts, shared
from modules.ui_components import InputAccordion

try:
    from backend.nn.flux import IntegratedFluxTransformer2DModel
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    
try:
    from backend.nn.unet import IntegratedUNet2DConditionModel, SpatialTransformer
    UNET_AVAILABLE = True
    SPATIAL_TRANSFORMER_AVAILABLE = True
except ImportError:
    UNET_AVAILABLE = False
    SPATIAL_TRANSFORMER_AVAILABLE = False

class TorchCompile(scripts.Script):
    original_flux_forward = None
    original_unet_forward = None
    original_spatial_transformer_forward = None
    compiled_models = {}
    compile_stats = {
        "total_compilations": 0, 
        "total_compile_time": 0.0,
        "successful_compilations": 0,
        "failed_compilations": 0,
        "cache_hits": 0,
        "compilation_infos": [],
        "model_types": {},
        "backend_usage": {},
        "mode_usage": {}
    }
    _logger = None
    
    @classmethod
    def get_logger(cls):
        """获取日志记录器"""
        if cls._logger is None:
            cls._logger = logging.getLogger("torch_compile_plugin")
            cls._logger.setLevel(logging.INFO)
            if not cls._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('[TorchCompile] %(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                cls._logger.addHandler(handler)
        return cls._logger
    
    def __init__(self):
        logger = self.get_logger()
        if FLUX_AVAILABLE and TorchCompile.original_flux_forward is None:
            TorchCompile.original_flux_forward = IntegratedFluxTransformer2DModel.inner_forward
            logger.info("已保存 Flux 模型原始 inner_forward ")
        if UNET_AVAILABLE and TorchCompile.original_unet_forward is None:
            TorchCompile.original_unet_forward = IntegratedUNet2DConditionModel.forward
            logger.info("已保存 UNet 模型原始 forward ")
        if SPATIAL_TRANSFORMER_AVAILABLE and TorchCompile.original_spatial_transformer_forward is None:
            TorchCompile.original_spatial_transformer_forward = SpatialTransformer.forward
            logger.info("已保存 SpatialTransformer 原始 forward")
        self._validate_model_compatibility()
    
    def _validate_model_compatibility(self):
        """验证模型兼容性"""
        logger = self.get_logger()
        torch_version = torch.__version__
        logger.info(f"PyTorch 版本: {torch_version}")
        
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile 不可用，需要 PyTorch 2.0+")
            return False
            
        try:
            available_backends = []
            for backend in ["inductor", "cudagraphs" ,"aot_eager", "nvfuser", "onnxrt", "tensorrt", "ipex"]:
                try:
                    test_fn = torch.compile(lambda x: x + 1, backend=backend)
                    test_fn(torch.tensor(1.0))
                    available_backends.append(backend)
                except Exception:
                    pass
            logger.info(f"可用编译后端: {available_backends}")
        except Exception as e:
            logger.warning(f"后端检测失败: {e}")
            
        if torch.cuda.is_available():
            logger.info(f"CUDA 可用，设备数量: {torch.cuda.device_count()}")
        else:
            logger.info("CUDA 不可用，将使用 CPU")           
        return True

    @staticmethod
    def _patched_spatial_transformer_forward(self_module, x, context=None, transformer_options={}):
        """
        这是用于替换 SpatialTransformer.forward 的新方法。
        它用 torch.compile 友好的 view 和 permute 操作替换了 einops.rearrange。
        """
        if not isinstance(context, list):
            context = [context] * len(self_module.transformer_blocks)
        
        b, c, h, w = x.shape
        x_in = x
        x = self_module.norm(x)

        if not self_module.use_linear:
            x = self_module.proj_in(x)

        x = x.view(b, x.shape[1], h * w).permute(0, 2, 1).contiguous()
        
        if self_module.use_linear:
            x = self_module.proj_in(x)

        for i, block in enumerate(self_module.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
            
        if self_module.use_linear:
            x = self_module.proj_out(x)

        x = x.permute(0, 2, 1).contiguous().view(b, x.shape[2], h, w)
        
        if not self_module.use_linear:
            x = self_module.proj_out(x)
            
        return x + x_in

    def _get_safe_compile_kwargs(self, compile_kwargs, model_type):
        """为不同模型类型获取安全的编译参数 - 改进版本"""
        safe_kwargs = compile_kwargs.copy()
        if model_type == "flux":
            if safe_kwargs.get('mode') == 'max-autotune':
                self.get_logger().info("Flux模型使用reduce-overhead模式以提高稳定性")
                safe_kwargs['mode'] = 'reduce-overhead'
                
        elif model_type == "unet":
            if safe_kwargs.get('mode') == 'max-autotune':
                self.get_logger().info("UNet模型从 'max-autotune' 切换到 'reduce-overhead' 以提高编译速度和稳定性")
                safe_kwargs['mode'] = 'reduce-overhead'
        else:
            self.get_logger().info("未知模型类型，使用保守编译设置")
            safe_kwargs['fullgraph'] = False
            safe_kwargs['mode'] = 'default'
            
        return safe_kwargs
    
    @staticmethod
    def _generate_model_hash(model, compile_kwargs):
        """生成模型和编译参数的哈希值"""
        try:
            model_info = {
                'model_id': id(model),
                'model_type': type(model).__name__,
                'model_config': getattr(model, 'config', {}) if hasattr(model, 'config') else {},
                'compile_kwargs': dict(sorted(compile_kwargs.items()))
            }
            try:
                if hasattr(model, 'state_dict'):
                    param_info = []
                    for name, param in list(model.state_dict().items())[:5]:
                        if hasattr(param, 'shape'):
                            param_info.append((name, tuple(param.shape)))
                    model_info['param_info'] = param_info
            except Exception:
                pass
            
            hash_str = str(sorted(model_info.items()))
            return hashlib.md5(hash_str.encode()).hexdigest()[:12]
        except Exception:
            simple_info = f"{id(model)}_{type(model).__name__}_{dict(sorted(compile_kwargs.items()))}"
            return hashlib.md5(simple_info.encode()).hexdigest()[:12]

    def title(self):
        return "Torch Compile加速"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # UI部分与您提供的代码完全一致，此处不重复
        with InputAccordion(False, label=self.title()) as enabled:
            torch_compile_available = hasattr(torch, 'compile')
            if not torch_compile_available:
                gr.Markdown("当前PyTorch版本不支持torch.compile，请升级到PyTorch 2.0+")
            with gr.Row():
                backend = gr.Dropdown(label="编译后端", choices=["inductor", "cudagraphs" ,"aot_eager", "nvfuser", "onnxrt", "tensorrt", "ipex"], value="inductor", scale=1, interactive=torch_compile_available)
                mode = gr.Dropdown(label="编译模式", choices=["default", "reduce-overhead", "max-autotune"], value="max-autotune", scale=1, interactive=torch_compile_available)
            with gr.Row():
                dynamic = gr.Checkbox(label="动态形状编译", value=True, info="允许不同的输入形状使用同一个编译模型", interactive=torch_compile_available)
                fullgraph = gr.Checkbox(label="完整图编译", value=False, info="强制编译整个计算图（可能会失败）", interactive=torch_compile_available)
            with gr.Row():
                compile_unet = gr.Checkbox(label="编译UNet/Transformer", value=True, info="编译主要的生成模型", interactive=torch_compile_available)
                auto_clear = gr.Checkbox(label="自动清理缓存", value=False, info="模型切换时自动清理编译缓存", interactive=torch_compile_available)
            with gr.Row():
                clear_cache = gr.Button(value="清除编译缓存", size="sm", interactive=torch_compile_available)
                show_stats = gr.Button(value="显示统计信息", size="sm", interactive=torch_compile_available)
            status = gr.Textbox(label="状态", value="未启用" if torch_compile_available else "PyTorch版本不支持", interactive=False)
            
            def clear_compile_cache():
                try:
                    count = len(TorchCompile.compiled_models)
                    TorchCompile.compiled_models.clear()
                    if hasattr(torch, '_dynamo'):
                        torch._dynamo.reset()
                    self.restore_original_methods()
                    return f"已清除{count}个编译缓存并恢复所有方法。"
                except Exception as e:
                    return f"清除缓存失败: {str(e)}"

            def show_compile_stats():
                try:
                    stats = TorchCompile.compile_stats
                    cache_count = len(TorchCompile.compiled_models)
                    total_time = stats.get("total_compile_time", 0.0)
                    total_compilations = stats.get("total_compilations", 0)
                    successful = stats.get("successful_compilations", 0)
                    failed = stats.get("failed_compilations", 0)
                    cache_hits = stats.get("cache_hits", 0)
                    success_rate = (successful / max(total_compilations, 1)) * 100
                    
                    result = f"## 编译统计信息\n\n"
                    result += f"**基本统计:**\n"
                    result += f"- 总编译次数: {total_compilations}\n"
                    result += f"- 成功编译: {successful} ({success_rate:.1f}%)\n"
                    result += f"- 失败编译: {failed}\n"
                    result += f"- 缓存命中: {cache_hits}\n"
                    result += f"- 当前缓存模型: {cache_count}个\n"
                    result += f"- 总编译时间: {total_time:.2f}秒\n"
                    result += f"- 平均编译时间: {total_time/max(successful, 1):.2f}秒\n\n"
                    
                    model_types = stats.get("model_types", {})
                    if model_types:
                        result += f"**模型类型统计:**\n"
                        for model_type, count in model_types.items():
                            result += f"- {model_type}: {count}次\n"
                        result += "\n"
                    
                    backend_usage = stats.get("backend_usage", {})
                    if backend_usage:
                        result += f"**后端使用统计:**\n"
                        for backend, count in backend_usage.items():
                            result += f"- {backend}: {count}次\n"
                        result += "\n"
                    
                    recent_infos = stats.get("compilation_infos", [])[-3:]
                    if recent_infos:
                        result += f"**最近的编译错误:**\n"
                        for i, info in enumerate(recent_infos, 1):
                            result += f"{i}. {info}\n"
                    return result
                except Exception as e:
                    return f"获取统计信息失败: {str(e)}"
            
            clear_cache.click(fn=clear_compile_cache, outputs=[status])
            show_stats.click(fn=show_compile_stats, outputs=[status])
        
        self.infotext_fields = [
            (enabled, lambda d: d.get("compile_enabled", False)),
            (backend, "compile_backend"),
            (mode, "compile_mode"),
            (dynamic, "compile_dynamic"),
            (fullgraph, "compile_fullgraph"),
            (compile_unet, "compile_compile_unet"),
            (auto_clear, "compile_auto_clear"),
        ]
        return [enabled, backend, mode, dynamic, fullgraph, compile_unet, auto_clear]

    def process(self, p, *args):
        enabled, backend, mode, dynamic, fullgraph, compile_unet, auto_clear = args
        self.restore_original_methods()
        if enabled and hasattr(torch, 'compile'):
            compile_kwargs = {'backend': backend, 'mode': mode, 'dynamic': dynamic, 'fullgraph': fullgraph}
            setattr(TorchCompile, "compile_kwargs", compile_kwargs)
            setattr(TorchCompile, "compile_unet", compile_unet)
            setattr(TorchCompile, "auto_clear", auto_clear)
            if compile_unet:
                self.apply_torch_compile(p, compile_kwargs)

    def apply_torch_compile(self, p, compile_kwargs):
        """应用torch.compile到模型 (最终修正版)"""
        logger = self.get_logger()
        
        # 1. 在编译前，先修补有问题的子模块
        if SPATIAL_TRANSFORMER_AVAILABLE and SpatialTransformer.forward is TorchCompile.original_spatial_transformer_forward:
            logger.info("为兼容torch.compile，正在修补 SpatialTransformer.forward...")
            SpatialTransformer.forward = self._patched_spatial_transformer_forward
        
        try:
            model_hash = self._generate_model_hash(p.sd_model, compile_kwargs)
            cache_key = f"{model_hash}"
            if cache_key in TorchCompile.compiled_models:
                logger.info("使用已编译的模型缓存")
                TorchCompile.compile_stats["cache_hits"] += 1
                return

            # 2. 恢复您正确的模型检测逻辑
            model_type = "unknown"
            is_flux = False
            flux_instances, unet_instances = [], []
            model_to_search = p.sd_model

            if hasattr(model_to_search, 'forge_objects'):
                logger.info("检测到 Forge 环境，正在遍历 forge_objects...")
                forge_objects = model_to_search.forge_objects
                modules_to_check = [
                    ('unet', getattr(forge_objects, 'unet', None)),
                    ('clip', getattr(forge_objects, 'clip', None)),
                    ('vae', getattr(forge_objects, 'vae', None)),
                    ('clipvision', getattr(forge_objects, 'clipvision', None)),
                ]
                for obj_name, obj in modules_to_check:
                    if obj is None: continue
                    modules_in_obj = []
                    if hasattr(obj, 'model') and hasattr(obj.model, 'named_modules'):
                         modules_in_obj = obj.model.named_modules()
                    elif hasattr(obj, 'named_modules'):
                         modules_in_obj = obj.named_modules()
                    for name, module in modules_in_obj:
                        if isinstance(module, IntegratedFluxTransformer2DModel):
                            flux_instances.append(module)
                        elif isinstance(module, IntegratedUNet2DConditionModel):
                            unet_instances.append(module)
            
            if flux_instances:
                model_type, is_flux = "flux", True
            elif unet_instances:
                model_type = "unet"
            else: # 如果在forge_objects中没找到，进行回退检查
                logger.warning("在 forge_objects 中未找到已知模型类型，将回退到基本检测。")
                if hasattr(p.sd_model, 'model') and hasattr(p.sd_model.model, 'diffusion_model'):
                    inner_model = p.sd_model.model.diffusion_model
                    if isinstance(inner_model, IntegratedFluxTransformer2DModel):
                        model_type, is_flux = "flux", True
                    elif isinstance(inner_model, IntegratedUNet2DConditionModel):
                        model_type = "unet"

            logger.info(f"检测到模型类型: {model_type}")
            
            safe_kwargs = self._get_safe_compile_kwargs(compile_kwargs, model_type)
            logger.info(f"开始编译模型 (后端: {safe_kwargs['backend']}, 模式: {safe_kwargs['mode']})")
            
            start_time = time.time()
            success = False
            if is_flux:
                success = self.compile_flux_model(p.sd_model, safe_kwargs)
            else:
                success = self.compile_unet_model(p.sd_model, safe_kwargs)
            setup_time = time.time() - start_time
            
            TorchCompile.compile_stats["total_compilations"] += 1
            if success:
                TorchCompile.compile_stats["successful_compilations"] += 1
                TorchCompile.compiled_models[cache_key] = True
                logger.info(f"模型编译设置成功，耗时: {setup_time:.3f}秒 (实际编译将在首次运行时发生)")
            else:
                TorchCompile.compile_stats["failed_compilations"] += 1
                logger.warning(f"模型编译设置失败，耗时: {setup_time:.3f}秒")
        except Exception as e:
            logger.error(f"模型编译过程中发生异常: {str(e)}\n{traceback.format_exc()}")
            TorchCompile.compile_stats["failed_compilations"] += 1
            self.restore_original_methods()

    def compile_flux_model(self, model, compile_kwargs):
        """编译Flux模型"""
        logger = self.get_logger()
        logger.warning("Flux模型编译功能尚未完全实现。")
        return False
        
    def compile_unet_model(self, model, compile_kwargs):
        """编译UNet (最终修正版)"""
        logger = self.get_logger()
        if not UNET_AVAILABLE:
            logger.warning("UNet模型不可用，跳过编译")
            return False

        try:
            target_function = TorchCompile.original_unet_forward
            compiled_fn = torch.compile(target_function, **compile_kwargs)
            IntegratedUNet2DConditionModel.forward = compiled_fn
            logger.info("UNet 模型 forward 类方法已成功替换为编译版本。")
            return True
        except Exception as e:
            logger.error(f"UNet模型编译失败: {e}\n{traceback.format_exc()}")
            self.restore_original_methods()
            return False

    def process_before_every_sampling(self, p, *args, **kwargs):
        """在每次采样前执行"""
        enabled = args[0] if args else False
        if not enabled:
            self.restore_original_methods()
        else:
            auto_clear = getattr(TorchCompile, "auto_clear", False)
            if auto_clear:
                current_model_id = id(p.sd_model)
                if hasattr(TorchCompile, "_last_model_id") and TorchCompile._last_model_id != current_model_id:
                    self.get_logger().info("检测到模型切换，自动清理编译缓存")
                    TorchCompile.compiled_models.clear()
                    if hasattr(torch, '_dynamo'):
                        torch._dynamo.reset()
                    self.restore_original_methods()
                TorchCompile._last_model_id = current_model_id

    def restore_original_methods(self):
        """恢复所有被修改过的原始方法。"""
        logger = self.get_logger()
        
        if SPATIAL_TRANSFORMER_AVAILABLE and TorchCompile.original_spatial_transformer_forward is not None:
            if SpatialTransformer.forward is not TorchCompile.original_spatial_transformer_forward:
                SpatialTransformer.forward = TorchCompile.original_spatial_transformer_forward
                logger.info("已恢复 SpatialTransformer 原始 forward")

        if UNET_AVAILABLE and TorchCompile.original_unet_forward is not None:
            if IntegratedUNet2DConditionModel.forward is not TorchCompile.original_unet_forward:
                IntegratedUNet2DConditionModel.forward = TorchCompile.original_unet_forward
                logger.info("已恢复 UNet 模型原始 forward")

        if FLUX_AVAILABLE and TorchCompile.original_flux_forward is not None:
            if IntegratedFluxTransformer2DModel.inner_forward is not TorchCompile.original_flux_forward:
                IntegratedFluxTransformer2DModel.inner_forward = TorchCompile.original_flux_forward
                logger.info("已恢复 Flux 模型原始 inner_forward")

    def postprocess(self, p, processed, *args):
        """后处理，确保在生成结束后恢复所有修改"""
        self.restore_original_methods()
