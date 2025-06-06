import torch
import sys
import time
import os
import traceback
import logging
import hashlib
import gradio as gr
from modules import scripts, shared
from modules.ui_components import InputAccordion

try:
    from backend.nn.flux import IntegratedFluxTransformer2DModel
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    
try:
    from backend.nn.unet import IntegratedUNet2DConditionModel
    UNET_AVAILABLE = True
except ImportError:
    UNET_AVAILABLE = False

class TorchCompile(scripts.Script):
    original_flux_forward = None
    original_unet_forward = None
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
        # 保存原始的forward
        if FLUX_AVAILABLE and TorchCompile.original_flux_forward is None:
            TorchCompile.original_flux_forward = IntegratedFluxTransformer2DModel.inner_forward
            self.get_logger().info("已保存 Flux 模型原始 inner_forward ")
        if UNET_AVAILABLE and TorchCompile.original_unet_forward is None:
            TorchCompile.original_unet_forward = IntegratedUNet2DConditionModel.forward
            self.get_logger().info("已保存 UNet 模型原始 forward ")
            
        # 初始化模型验证
        self._validate_model_compatibility()
    
    def _validate_model_compatibility(self):
        """验证模型兼容性"""
        logger = self.get_logger()
        torch_version = torch.__version__
        logger.info(f"PyTorch 版本: {torch_version}")
        
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile 不可用，需要 PyTorch 2.0+")
            return False
            
        # 检查可用的编译后端
        try:
            available_backends = []
            for backend in ["inductor", "cudagraphs" ,"aot_eager", "nvfuser", "onnxrt", "tensorrt", "ipex"]:
                try:
                    # 测试编译
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
    def _get_safe_compile_kwargs(self, compile_kwargs, model_type):
        """为不同模型类型获取安全的编译参数 - 改进版本"""
        safe_kwargs = compile_kwargs.copy()
        if model_type == "flux":
            if safe_kwargs.get('mode') == 'max-autotune':
                self.get_logger().info("Flux模型使用reduce-overhead模式以提高稳定性")
                safe_kwargs['mode'] = 'reduce-overhead'
                
        elif model_type == "unet":
            pass        
        else:
            self.get_logger().info("未知模型类型，使用保守编译设置")
            safe_kwargs['fullgraph'] = False
            safe_kwargs['mode'] = 'default'
            
        return safe_kwargs
    
    @staticmethod
    def _generate_model_hash(model, compile_kwargs):
        """生成模型和编译参数的哈希值 - 改进版"""
        try:
            model_filename = "unknown_model"
            # 优先获取模型文件名作为唯一标识
            if hasattr(shared, 'sd_model') and hasattr(shared.sd_model, 'sd_checkpoint_info') and hasattr(shared.sd_model.sd_checkpoint_info, 'filename'):
                model_filename = os.path.basename(shared.sd_model.sd_checkpoint_info.filename)

            model_info = {
                'model_filename': model_filename, # 使用文件名
                'model_type': type(model).__name__,
                # 将字典转换为排序后的字符串，保证哈希一致性
                'compile_kwargs': str(sorted(compile_kwargs.items()))
            }

            hash_str = str(sorted(model_info.items()))
            return hashlib.md5(hash_str.encode()).hexdigest()[:12]
        except Exception as e:
            logger = TorchCompile.get_logger()
            logger.warning(f"无法生成基于文件名的哈希，回退到简单哈希: {e}")
            simple_info = f"{id(model)}_{type(model).__name__}_{str(sorted(compile_kwargs.items()))}"
            return hashlib.md5(simple_info.encode()).hexdigest()[:12]

    def title(self):
        return "Torch Compile加速"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            # 检查torch.compile可用性
            torch_compile_available = hasattr(torch, 'compile')
            
            if not torch_compile_available:
                gr.Markdown("当前PyTorch版本不支持torch.compile，请升级到PyTorch 2.0+")
            
            with gr.Row():
                backend = gr.Dropdown(
                    label="编译后端", 
                    choices=["inductor", "cudagraphs" ,"aot_eager", "nvfuser", "onnxrt", "tensorrt", "ipex"], 
                    value="inductor",
                    scale=1,
                    interactive=torch_compile_available
                )
                mode = gr.Dropdown(
                    label="编译模式", 
                    choices=["default", "reduce-overhead", "max-autotune"], 
                    value="max-autotune",
                    scale=1,
                    interactive=torch_compile_available
                )
            
            with gr.Row():
                dynamic = gr.Checkbox(
                    label="动态形状编译", 
                    value=True,
                    info="允许不同的输入形状使用同一个编译模型",
                    interactive=torch_compile_available
                )
                fullgraph = gr.Checkbox(
                    label="完整图编译", 
                    value=False,
                    info="强制编译整个计算图（可能会失败）",
                    interactive=torch_compile_available
                )
            
            with gr.Row():
                compile_unet = gr.Checkbox(
                    label="编译UNet/Transformer", 
                    value=True,
                    info="编译主要的生成模型",
                    interactive=torch_compile_available
                )
                auto_clear = gr.Checkbox(
                    label="自动清理缓存", 
                    value=False,
                    info="模型切换时自动清理编译缓存",
                    interactive=torch_compile_available
                )
            
            with gr.Row():
                clear_cache = gr.Button(
                    value="清除编译缓存",
                    size="sm",
                    interactive=torch_compile_available
                )
                show_stats = gr.Button(
                    value="显示统计信息",
                    size="sm",
                    interactive=torch_compile_available
                )
            
            status = gr.Textbox(
                label="状态", 
                value="未启用" if torch_compile_available else "PyTorch版本不支持", 
                interactive=False
            )
            
            # 清除缓存按钮的回调
            def clear_compile_cache():
                try:
                    count = len(TorchCompile.compiled_models)
                    TorchCompile.compiled_models.clear()
                    if hasattr(torch, '_dynamo'):
                        torch._dynamo.reset()
                    return f"已清除{count}个编译缓存"
                except Exception as e:
                    return f"清除缓存失败: {str(e)}"
            
            # 显示统计信息的回调
            def show_compile_stats():
                try:
                    stats = TorchCompile.compile_stats
                    cache_count = len(TorchCompile.compiled_models)
                    total_time = stats.get("total_compile_time", 0.0)
                    total_compilations = stats.get("total_compilations", 0)
                    successful = stats.get("successful_compilations", 0)
                    failed = stats.get("failed_compilations", 0)
                    cache_hits = stats.get("cache_hits", 0)
                    
                    # 计算成功率
                    success_rate = (successful / max(total_compilations, 1)) * 100
                    
                    # 构建详细统计信息
                    result = f"## 编译统计信息\n\n"
                    result += f"**基本统计:**\n"
                    result += f"- 总编译次数: {total_compilations}\n"
                    result += f"- 成功编译: {successful} ({success_rate:.1f}%)\n"
                    result += f"- 失败编译: {failed}\n"
                    result += f"- 缓存命中: {cache_hits}\n"
                    result += f"- 当前缓存模型: {cache_count}个\n"
                    result += f"- 总编译时间: {total_time:.2f}秒\n"
                    result += f"- 平均编译时间: {total_time/max(total_compilations, 1):.2f}秒\n\n"
                    
                    # 模型类型统计
                    model_types = stats.get("model_types", {})
                    if model_types:
                        result += f"**模型类型统计:**\n"
                        for model_type, count in model_types.items():
                            result += f"- {model_type}: {count}次\n"
                        result += "\n"
                    
                    # 后端使用统计
                    backend_usage = stats.get("backend_usage", {})
                    if backend_usage:
                        result += f"**后端使用统计:**\n"
                        for backend, count in backend_usage.items():
                            result += f"- {backend}: {count}次\n"
                        result += "\n"
                    
                    # 最近的错误
                    recent_infos = stats.get("compilation_infos", [])[-3:]
                    if recent_infos:
                        result += f"**最近的编译错误:**\n"
                        for i, info in enumerate(recent_infos, 1):
                            result += f"{i}. {info}\n"
                    
                    return result
                    
                except Exception as e:
                    return f"获取统计信息失败: {str(e)}"
            
            clear_cache.click(
                fn=clear_compile_cache,
                outputs=[status]
            )
            
            show_stats.click(
                fn=show_compile_stats,
                outputs=[status]
            )
        
        # 设置不保存到配置
        enabled.do_not_save_to_config = True
        backend.do_not_save_to_config = True
        mode.do_not_save_to_config = True
        dynamic.do_not_save_to_config = True
        fullgraph.do_not_save_to_config = True
        compile_unet.do_not_save_to_config = True
        auto_clear.do_not_save_to_config = True

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

        if enabled:
            if not hasattr(torch, 'compile'):
                print("警告: 当前PyTorch版本不支持torch.compile，请升级到PyTorch 2.0+")
                return

            # 设置编译参数
            compile_kwargs = {
                'backend': backend,
                'mode': mode,
                'dynamic': dynamic,
                'fullgraph': fullgraph
            }

            # 保存参数到额外生成参数中
            '''
            p.extra_generation_params.update({
                "compile_enabled": enabled,
                "compile_backend": backend,
                "compile_mode": mode,
                "compile_dynamic": dynamic,
                "compile_fullgraph": fullgraph,
                "compile_compile_unet": compile_unet,
                "compile_auto_clear": auto_clear,
            })'''

            # 设置类属性供其他使用
            setattr(TorchCompile, "compile_kwargs", compile_kwargs)
            setattr(TorchCompile, "compile_unet", compile_unet)
            setattr(TorchCompile, "auto_clear", auto_clear)

            # 应用编译
            if compile_unet:
                self.apply_torch_compile(p, compile_kwargs)

    def apply_torch_compile(self, p, compile_kwargs):
        """应用torch.compile到模型"""
        logger = self.get_logger()
        
        try:
            # 生成缓存键
            model_hash = self._generate_model_hash(p.sd_model, compile_kwargs)
            cache_key = f"{model_hash}_{compile_kwargs['backend']}_{compile_kwargs['mode']}"
            
            # 检查是否已经编译过这个模型
            if cache_key in TorchCompile.compiled_models:
                logger.info("使用已编译的模型缓存")
                TorchCompile.compile_stats["cache_hits"] += 1
                return
            backend = compile_kwargs['backend']
            mode = compile_kwargs['mode']
            TorchCompile.compile_stats["backend_usage"][backend] = TorchCompile.compile_stats["backend_usage"].get(backend, 0) + 1
            TorchCompile.compile_stats["mode_usage"][mode] = TorchCompile.compile_stats["mode_usage"].get(mode, 0) + 1
            # 检测模型类型
            model_type = "unknown"
            is_flux = False
            
            # 直接检查模型中是否有IntegratedFluxTransformer2DModel实例
            flux_instances = []
            unet_instances = []
            
            # 深度遍历模型查找目标类型
            # 需要通过 forge_objects 访问实际的 PyTorch 模块
            model_to_search = p.sd_model
            if hasattr(p.sd_model, 'forge_objects'):
                # 搜索 forge_objects 中的模块
                forge_objects = p.sd_model.forge_objects
                modules_to_check = []
                
                # 收集所有 forge_objects 中的模块
                if hasattr(forge_objects, 'unet') and forge_objects.unet is not None:
                    modules_to_check.append(('unet', forge_objects.unet))
                if hasattr(forge_objects, 'clip') and forge_objects.clip is not None:
                    modules_to_check.append(('clip', forge_objects.clip))
                if hasattr(forge_objects, 'vae') and forge_objects.vae is not None:
                    modules_to_check.append(('vae', forge_objects.vae))
                if hasattr(forge_objects, 'clipvision') and forge_objects.clipvision is not None:
                    modules_to_check.append(('clipvision', forge_objects.clipvision))
                
                # 搜索每个模块
                for obj_name, obj in modules_to_check:
                    if hasattr(obj, 'model') and hasattr(obj.model, 'named_modules'):
                        for name, module in obj.model.named_modules():
                            full_name = f"{obj_name}.model.{name}"
                            if isinstance(module, IntegratedFluxTransformer2DModel):
                                flux_instances.append((full_name, module))
                            elif isinstance(module, IntegratedUNet2DConditionModel):
                                unet_instances.append((full_name, module))
                    elif hasattr(obj, 'named_modules'):
                        for name, module in obj.named_modules():
                            full_name = f"{obj_name}.{name}"
                            if isinstance(module, IntegratedFluxTransformer2DModel):
                                flux_instances.append((full_name, module))
                            elif isinstance(module, IntegratedUNet2DConditionModel):
                                unet_instances.append((full_name, module))
            else:
                # 如果没有 forge_objects，尝试直接访问（用于其他类型的模型）
                if hasattr(model_to_search, 'named_modules'):
                    try:
                        for name, module in model_to_search.named_modules():
                            if isinstance(module, IntegratedFluxTransformer2DModel):
                                flux_instances.append((name, module))
                            elif isinstance(module, IntegratedUNet2DConditionModel):
                                unet_instances.append((name, module))
                    except Exception as e:
                        logger.info(f"搜索标准模型模块失败: {e}")
                else:
                    logger.info("模型没有 named_modules ，无法遍历模块")
            
            # 根据找到的实例确定模型类型
            if flux_instances:
                model_type = "flux"
                is_flux = True
                logger.info(f"检测到{len(flux_instances)}个Flux实例: {[name for name, _ in flux_instances]}")
            elif unet_instances:
                model_type = "unet"
                logger.info(f"检测到{len(unet_instances)}个UNet实例: {[name for name, _ in unet_instances]}")
            else:
                # 回退到类名检测
                model_class_name = type(p.sd_model).__name__
                if 'flux' in model_class_name.lower() or 'transformer' in model_class_name.lower():
                    model_type = "flux"
                    is_flux = True
                    logger.info(f"通过类名检测到Flux模型: {model_class_name}")
                elif 'unet' in model_class_name.lower():
                    model_type = "unet"
                    logger.info(f"通过类名检测到UNet模型: {model_class_name}")
                else:
                    logger.warning(f"未知模型类型: {model_class_name}")
                    # 最后的回退检测
                    if hasattr(p.sd_model, 'model') and hasattr(p.sd_model.model, 'diffusion_model'):
                        inner_model = p.sd_model.model.diffusion_model
                        inner_class_name = type(inner_model).__name__
                        if isinstance(inner_model, IntegratedFluxTransformer2DModel):
                            model_type = "flux"
                            is_flux = True
                            logger.info(f"在diffusion_model中检测到Flux: {inner_class_name}")
                        elif isinstance(inner_model, IntegratedUNet2DConditionModel):
                            model_type = "unet"
                            logger.info(f"在diffusion_model中检测到UNet: {inner_class_name}")
                    
            TorchCompile.compile_stats["model_types"][model_type] = TorchCompile.compile_stats["model_types"].get(model_type, 0) + 1
            
            logger.info(f"开始编译模型 (类型: {model_type}, 后端: {backend}, 模式: {mode})")
            
            # 记录编译开始时间
            start_time = time.time()
            
            # 根据模型类型应用编译
            if is_flux:
                success = self.compile_flux_model(p.sd_model, compile_kwargs)
            else:
                success = self.compile_unet_model(p.sd_model, compile_kwargs)
            # 记录编译设置时间和结果
            setup_time = time.time() - start_time
            TorchCompile.compile_stats["total_compilations"] += 1
            
            if success:
                TorchCompile.compile_stats["successful_compilations"] += 1
                TorchCompile.compiled_models[cache_key] = {
                    "timestamp": time.time(),
                    "model_type": model_type,
                    "backend": backend,
                    "mode": mode,
                    "setup_time": setup_time
                }
                logger.info(f"模型编译设置成功，耗时: {setup_time:.3f}秒 (真正的编译将在首次运行时发生)")
            else:
                TorchCompile.compile_stats["failed_compilations"] += 1
                logger.warning(f"模型编译设置失败，耗时: {setup_time:.3f}秒")
        except Exception as e:
            logger.info(f"模型编译过程中发生异常: {str(e)}")
            logger.info(f"异常详情: {traceback.format_exc()}")
            
            # 记录错误
            info_msg = f"{type(e).__name__}: {str(e)}"
            TorchCompile.compile_stats["compilation_infos"].append(info_msg)
            TorchCompile.compile_stats["failed_compilations"] += 1
            
            # 恢复原始
            self.restore_original_methods()
            logger.info("已回退到原始模型")

    def compile_flux_model(self, model, compile_kwargs):
        """编译Flux模型"""
        logger = self.get_logger()
        
        if not FLUX_AVAILABLE:
            logger.warning("Flux模型不可用，跳过编译")
            return False
        
        # 检查是否有可编译的Flux模型实例
        flux_instances = []
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
            if isinstance(diffusion_model, IntegratedFluxTransformer2DModel):
                flux_instances.append(diffusion_model)          # 如果没有找到直接的实例，查找所有IntegratedFluxTransformer2DModel实例
        if not flux_instances:
            if hasattr(model, 'forge_objects'):
                forge_objects = model.forge_objects
                if hasattr(forge_objects, 'unet') and forge_objects.unet is not None:
                    if hasattr(forge_objects.unet, 'model') and hasattr(forge_objects.unet.model, 'named_modules'):
                        try:
                            for name, module in forge_objects.unet.model.named_modules():
                                if isinstance(module, IntegratedFluxTransformer2DModel):
                                    flux_instances.append(module)
                                    logger.info(f"找到Flux模型实例: unet.model.{name}")
                        except Exception as e:
                            logger.info(f"搜索 forge_objects.unet.model 失败: {e}")
                    elif hasattr(forge_objects.unet, 'named_modules'):
                        try:
                            for name, module in forge_objects.unet.named_modules():
                                if isinstance(module, IntegratedFluxTransformer2DModel):
                                    flux_instances.append(module)
                                    logger.info(f"找到Flux模型实例: unet.{name}")
                        except Exception as e:
                            logger.info(f"搜索 forge_objects.unet 失败: {e}")
                
                for component_name in ['clip', 'vae', 'clipvision']:
                    if hasattr(forge_objects, component_name):
                        component = getattr(forge_objects, component_name)
                        if component is not None:
                            if hasattr(component, 'model') and hasattr(component.model, 'named_modules'):
                                try:
                                    for name, module in component.model.named_modules():
                                        if isinstance(module, IntegratedFluxTransformer2DModel):
                                            flux_instances.append(module)
                                            logger.info(f"找到Flux模型实例: {component_name}.model.{name}")
                                except Exception as e:
                                    logger.info(f"搜索 forge_objects.{component_name}.model 失败: {e}")
                            elif hasattr(component, 'named_modules'):
                                try:
                                    for name, module in component.named_modules():
                                        if isinstance(module, IntegratedFluxTransformer2DModel):
                                            flux_instances.append(module)
                                            logger.info(f"找到Flux模型实例: {component_name}.{name}")
                                except Exception as e:
                                    logger.info(f"搜索 forge_objects.{component_name} 失败: {e}")
            elif hasattr(model, 'named_modules'):
                # 对于标准 PyTorch 模型
                try:
                    for name, module in model.named_modules():
                        if isinstance(module, IntegratedFluxTransformer2DModel):
                            flux_instances.append(module)
                            logger.info(f"找到Flux模型实例: {name}")
                except Exception as e:
                    logger.info(f"搜索标准模型失败: {e}")
            else:
                logger.info("模型既没有 forge_objects 也没有 named_modules ")
        
        if not flux_instances:
            logger.warning("未找到可编译的IntegratedFluxTransformer2DModel实例")
            return False
            
        try:
            def compiled_inner_forward(self, *args, **kwargs):
                # 创建编译版本的forward
                if not hasattr(self, '_compiled_forward'):
                    try:
                        logger.info(f"为 Flux 模型创建编译版本，参数: {compile_kwargs}")
                        if not self._validate_flux_inputs(*args, **kwargs):
                            logger.warning("Flux 输入验证失败，使用原始")
                            return TorchCompile.original_flux_forward(self, *args, **kwargs)
                        
                        safe_compile_kwargs = self._get_safe_compile_kwargs(compile_kwargs, "flux")
                        
                        # 尝试编译，如果失败则使用更保守的设置
                        try:
                            self._compiled_forward = torch.compile(
                                TorchCompile.original_flux_forward, 
                                **safe_compile_kwargs
                            )
                            logger.info("Flux 模型编译函数创建成功")
                        except Exception as compile_info:
                            logger.warning(f"Flux完整编译失败，尝试保守设置: {compile_info}")
                            # 使用最保守的设置重试
                            fallback_kwargs = safe_compile_kwargs.copy()
                            fallback_kwargs['fullgraph'] = False
                            fallback_kwargs['mode'] = 'default'
                            
                            self._compiled_forward = torch.compile(
                                TorchCompile.original_flux_forward, 
                                **fallback_kwargs
                            )
                            logger.info("Flux 模型使用保守设置编译成功")
                        
                    except Exception as e:
                        logger.info(f"Flux模型编译失败: {str(e)}")
                        logger.info(f"详细错误: {traceback.format_exc()}")
                        # 回退到原始
                        return TorchCompile.original_flux_forward(self, *args, **kwargs)
                
                try:
                    return self._compiled_forward(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"编译版本执行失败，回退到原始: {str(e)}")
                    logger.error(f"详细错误: {traceback.format_exc()}")
                    return TorchCompile.original_flux_forward(self, *args, **kwargs)
            # 添加输入验证
            def _validate_flux_inputs(self, *args, **kwargs):
                """验证 Flux 模型输入 - 基于实际的inner_forward签名"""
                try:
                    # inner_forward: (img, img_ids, txt, txt_ids, timesteps, y, guidance=None)
                    if len(args) < 6:  # 需要 img, img_ids, txt, txt_ids, timesteps, y
                        logger.info(f"Flux输入参数不足: {len(args)} < 6")
                        return False
                    
                    img, img_ids, txt, txt_ids, timesteps, y = args[:6]
                    
                    # 验证必需的张量参数
                    required_tensors = [img, img_ids, txt, txt_ids, timesteps, y]
                    for i, tensor in enumerate(required_tensors):
                        if not isinstance(tensor, torch.Tensor):
                            logger.info(f"Flux参数{i}不是张量: {type(tensor)}")
                            return False
                    
                    # 验证张量维度
                    if img.ndim != 3 or txt.ndim != 3:
                        logger.info(f"Flux img或txt维度错误: img={img.ndim}, txt={txt.ndim}")
                        return False
                    
                    # 验证设备一致性
                    device = img.device
                    for tensor in required_tensors:
                        if tensor.device != device:
                            logger.info("Flux张量设备不一致")
                            return False
                    
                    # 检查CUDA性能警告
                    if not device.type == 'cuda' and torch.cuda.is_available():
                        logger.info("检测到CPU张量，可能影响Flux编译性能")
                    
                    return True
                except Exception as e:
                    logger.info(f"Flux输入验证异常: {e}")
                    return False
            
            # 绑定验证
            IntegratedFluxTransformer2DModel._validate_flux_inputs = _validate_flux_inputs
            
            # 替换inner_forward
            IntegratedFluxTransformer2DModel.inner_forward = compiled_inner_forward
            logger.info("Flux 模型 inner_forward 已替换为编译版本")
            return True
            
        except Exception as e:
            logger.info(f"Flux 模型编译设置失败: {str(e)}")
            return False
    def compile_unet_model(self, model, compile_kwargs):
        """编译UNet"""
        logger = self.get_logger()
        if not UNET_AVAILABLE:
            logger.warning("UNet模型不可用，跳过编译")
            return False
            
        try:
            # 用于跟踪编译状态的变量
            compilation_info = {
                'setup_time': 0.0,
                'first_run_time': 0.0,
                'compiled': False,
                'setup_completed': False
            }
            def compiled_forward(self, *args, **kwargs):
                #logger.info("compiled_forward 被调用了！")
                if not hasattr(self, '_compiled_forward'):
                    setup_start = time.time()
                    try:
                        logger.info(f"为 UNet 模型创建编译版本，参数: {compile_kwargs}")
                        self._compiled_forward = torch.compile(
                            TorchCompile.original_unet_forward, 
                            **compile_kwargs
                        )
                        
                        setup_time = time.time() - setup_start
                        compilation_info['setup_time'] = setup_time
                        compilation_info['setup_completed'] = True
                        
                        logger.info(f"UNet 模型编译装饰器创建成功 (设置耗时: {setup_time:.3f}秒)")
                        logger.info("注意：真正的编译将在模型首次推理时发生，可能需要额外的时间")
                        
                    except Exception as e:
                        logger.info(f"UNet模型编译设置失败: {str(e)}")
                        logger.info(f"详细错误: {traceback.format_exc()}")                    
                        return TorchCompile.original_unet_forward(self, *args, **kwargs)
                
                try:
                    # 检查是否是第一次运行（真正的编译会在这里发生）
                    is_first_run = not compilation_info['compiled']
                    if is_first_run:
                        logger.info("开始第一次运行 UNet 模型（真正的编译将在此时发生）...")
                        first_run_start = time.time()
                    
                    # 添加调试信息
                    #logger.info(f"调用编译版本 UNet forward，首次运行: {is_first_run}")
                    result = self._compiled_forward(self, *args, **kwargs)
                    
                    if is_first_run:
                        first_run_time = time.time() - first_run_start
                        compilation_info['first_run_time'] = first_run_time
                        compilation_info['compiled'] = True
                        
                        total_time = compilation_info['setup_time'] + first_run_time
                        logger.info(f"UNet 模型真正编译完成！")
                        logger.info(f"  - 设置时间: {compilation_info['setup_time']:.3f}秒")
                        logger.info(f"  - 首次运行(编译)时间: {first_run_time:.3f}秒")
                        logger.info(f"  - 总计时间: {total_time:.3f}秒")
                        
                        # 更新全局统计信息
                        TorchCompile.compile_stats["total_compile_time"] += first_run_time
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"编译版本执行失败，回退到原始: {str(e)}")
                    return TorchCompile.original_unet_forward(self, *args, **kwargs)
            
            # 添加输入验证
            def _validate_unet_inputs(self, *args, **kwargs):
                """验证 UNet 模型输入 - 基于实际的forward签名"""
                try:
                    # forward的签名: (x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs)
                    if len(args) < 1:  # x 是必需的
                        logger.info("UNet缺少必需的x参数")
                        return False
                    
                    x = args[0]
                    if not isinstance(x, torch.Tensor):
                        logger.info(f"UNet的x不是张量: {type(x)}")
                        return False
                    
                    # 验证张量维度 - UNet通常期望4D张量 (B, C, H, W)
                    if x.dim() < 3:
                        logger.info(f"UNet输入维度过低: {x.dim()}")
                        return False
                    
                    # 检查批量大小
                    if x.shape[0] == 0:
                        logger.info("UNet输入批量大小为0")
                        return False
                    
                    # 验证可选参数
                    if len(args) > 1:
                        timesteps = args[1] if len(args) > 1 else kwargs.get('timesteps')
                        if timesteps is not None and not isinstance(timesteps, torch.Tensor):
                            logger.info(f"UNet timesteps类型错误: {type(timesteps)}")
                            return False
                    
                    if not x.device.type == 'cuda' and torch.cuda.is_available():
                        logger.info("检测到CPU张量，可能影响UNet编译性能")
                    
                    return True
                except Exception as e:
                    logger.info(f"UNet输入验证异常: {e}")
                    return False
            
            # 绑定验证
            IntegratedUNet2DConditionModel._validate_unet_inputs = _validate_unet_inputs
            
            # 替换forward
            IntegratedUNet2DConditionModel.forward = compiled_forward
            logger.info("UNet 模型 forward 已替换为编译版本")
            return True
            
        except Exception as e:
            logger.info(f"UNet 模型编译设置失败: {str(e)}")
            return False

    def process_before_every_sampling(self, p, *args, **kwargs):
        """在每次采样前执行"""
        logger = self.get_logger()
        enabled = args[0] if args else False
        
        if not enabled:
            # 如果未启用，恢复原始
            self.restore_original_methods()
        else:
            # 检查是否需要自动清理缓存
            auto_clear = getattr(TorchCompile, "auto_clear", False)
            if auto_clear:
                current_model_id = id(p.sd_model)
                # 如果模型ID改变了，清理旧的编译缓存
                if hasattr(TorchCompile, "_last_model_id") and TorchCompile._last_model_id != current_model_id:
                    cache_count = len(TorchCompile.compiled_models)
                    logger.info(f"检测到模型切换，自动清理 {cache_count} 个编译缓存")
                    TorchCompile.compiled_models.clear()
                    
                    # 重置 dynamo 缓存
                    try:
                        if hasattr(torch, '_dynamo'):
                            torch._dynamo.reset()
                        logger.info("Dynamo 缓存已重置")
                    except Exception as e:
                        logger.warning(f"重置 Dynamo 缓存失败: {e}")
                        
                TorchCompile._last_model_id = current_model_id

    def restore_original_methods(self):
        """恢复原始的forward"""
        logger = self.get_logger()
        
        if FLUX_AVAILABLE and TorchCompile.original_flux_forward is not None:
            IntegratedFluxTransformer2DModel.inner_forward = TorchCompile.original_flux_forward
            logger.info("已恢复 Flux 模型原始 inner_forward ")
        if UNET_AVAILABLE and TorchCompile.original_unet_forward is not None:
            IntegratedUNet2DConditionModel.forward = TorchCompile.original_unet_forward
            logger.info("已恢复 UNet 模型原始 forward ")

    def postprocess(self, p, processed, *args):
        """后处理，清理资源"""
        self.restore_original_methods()