"""
RealPushT Dataset Visualizer
可视化工具用于查看RealPushT数据集的RGB图像序列及其对应的状态和动作信息
"""

import os
import cv2
import numpy as np
import zarr
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from dataclasses import dataclass
from filelock import FileLock
from omegaconf import OmegaConf
import threading

# 在文件开头添加正确的导入
from diffusion_policy.common.replay_buffer import ReplayBuffer
# from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
# register_codecs()

# 设置浮点数显示格式
np.set_printoptions(precision=3, suppress=True)

@dataclass
class VisualizationConfig:
    """可视化配置"""
    dataset_path: str = "data/pusht_real/real_pusht_20230105"
    episode_idx: int = 0
    start_step: int = 0
    num_frames: Optional[int] = None
    fps: int = 10
    scale: float = 1.0
    show_info: bool = True
    save_video: bool = False
    video_output_path: str = "visualization_output.mp4"
    cache_enabled: bool = True

class RealPushTVisualizer:
    """RealPushT数据集可视化器"""
    
    def __init__(self, config: VisualizationConfig):
        # 在初始化时注册编解码器（关键修改！）
        from diffusion_policy.common.imagecodecs_numcodecs import register_codecs
        register_codecs()
        self.config = config
        self.replay_buffer = None
        self.episode_data = {}
        self.current_step = config.start_step
        self.paused = False
        self.quit_requested = False
        self.video_writer = None
        self.info_panel_height = 0
        
        # 形状元数据（根据您的配置文件）
        self.shape_meta = {
            'obs': {
                'camera_0': {'shape': [3, 240, 320], 'type': 'rgb'},
                'camera_1': {'shape': [3, 240, 320], 'type': 'rgb'},
                'robot_eef_pose': {'shape': [6], 'type': 'low_dim'},
                'robot_gripper': {'shape': [1], 'type': 'low_dim'},
                'stage': {'shape': [1], 'type': 'low_dim'}
            },
            'action': {'shape': [7]}
        }
        
        # 创建颜色映射用于可视化
        self.colors = {
            'text': (255, 255, 255),      # 白色
            'background': (40, 40, 40),    # 深灰色
            'highlight': (0, 255, 255),    # 青色
            'progress_bar': (0, 255, 0),   # 绿色
            'progress_bg': (100, 100, 100) # 灰色
        }
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载数据集"""
        print(f"加载数据集: {self.config.dataset_path}")
        
        # 检查数据集路径
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        # 使用缓存（改进版）
        if self.config.cache_enabled:
            # fingerprint shape_meta（与原有代码保持一致）
            shape_meta_json = json.dumps(self.shape_meta, sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = dataset_path / f"visualization_cache_{shape_meta_hash}.zarr.zip"
            cache_lock_path = Path(str(cache_zarr_path) + '.lock')
            
            print(f"缓存路径: {cache_zarr_path}")
            
            try:
                # 使用FileLock锁定缓存文件（与原有代码保持一致）
                with FileLock(str(cache_lock_path)):
                    if not cache_zarr_path.exists():
                        # 缓存不存在，创建缓存
                        print('缓存不存在，创建缓存...')
                        self._load_from_raw_data(dataset_path)
                        print('保存缓存到磁盘...')
                        with zarr.ZipStore(str(cache_zarr_path), mode='w') as zip_store:
                            self.replay_buffer.save_to_store(store=zip_store)
                        print('缓存创建成功!')
                    else:
                        # 加载缓存
                        print('加载缓存的ReplayBuffer...')
                        with zarr.ZipStore(str(cache_zarr_path), mode='r') as zip_store:
                            self.replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, 
                                store=zarr.MemoryStore()
                            )
                        
                        # 验证缓存数据（新增）
                        if self._validate_replay_buffer():
                            print('缓存加载成功!')
                        else:
                            print('警告: 缓存数据无效，重新创建...')
                            # 删除损坏的缓存
                            cache_zarr_path.unlink(missing_ok=True)
                            # 重新加载原始数据
                            self._load_from_raw_data(dataset_path)
                            # 重新保存缓存
                            with zarr.ZipStore(str(cache_zarr_path), mode='w') as zip_store:
                                self.replay_buffer.save_to_store(store=zip_store)
                            print('缓存重新创建成功!')
            
            except Exception as e:
                print(f"缓存操作失败: {e}")
                print("尝试从原始数据加载...")
                # 删除可能损坏的缓存文件
                cache_zarr_path.unlink(missing_ok=True)
                cache_lock_path.unlink(missing_ok=True)
                # 从原始数据加载
                self._load_from_raw_data(dataset_path)
        
        else:
            # 不使用缓存
            print("禁用缓存，直接从原始数据加载...")
            self._load_from_raw_data(dataset_path)


    def _validate_replay_buffer(self) -> bool:
        """验证ReplayBuffer数据是否有效"""
        if self.replay_buffer is None:
            return False
        
        # 检查基本属性
        if not hasattr(self.replay_buffer, 'n_steps'):
            return False
        if not hasattr(self.replay_buffer, 'n_episodes'):
            return False
        if not hasattr(self.replay_buffer, 'episode_ends'):
            return False
        if not hasattr(self.replay_buffer, 'data'):
            return False
        
        # 检查数据是否为空
        if self.replay_buffer.n_steps <= 0:
            return False
        if self.replay_buffer.n_episodes <= 0:
            return False
        if len(self.replay_buffer.episode_ends) <= 0:
            return False
        if len(self.replay_buffer.data) <= 0:
            return False
        
        # 检查关键数据键是否存在
        required_keys = ['action']
        for key in required_keys:
            if key not in self.replay_buffer.data:
                print(f"警告: 缺少关键数据键: {key}")
                return False
        
        return True
    def _get_cache_path(self) -> Path:
        """获取缓存路径"""
        dataset_path = Path(self.config.dataset_path)
        shape_meta_json = json.dumps(self.shape_meta, sort_keys=True)
        shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
        return dataset_path / f"visualization_cache_{shape_meta_hash}.zarr.zip"
    
    def _load_from_raw_data(self, dataset_path: Path):
        """从原始数据加载"""
        try:
            # 使用线程控制（与原有代码保持一致）
            cv2.setNumThreads(1)
            
            # 这里需要导入实际的real_data_to_replay_buffer函数
            from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
            
            # 提取RGB键和低维键
            rgb_keys = []
            lowdim_keys = []
            out_resolutions = {}
            
            for key, attr in self.shape_meta['obs'].items():
                if attr['type'] == 'rgb':
                    rgb_keys.append(key)
                    c, h, w = attr['shape']
                    out_resolutions[key] = (w, h)
                else:
                    lowdim_keys.append(key)
            
            # 添加动作键
            lowdim_keys.append('action')
            
            # 加载数据
            store = zarr.MemoryStore()
            
            print(f"加载图像键: {rgb_keys}")
            print(f"加载低维键: {lowdim_keys}")
            
            self.replay_buffer = real_data_to_replay_buffer(
                dataset_path=str(dataset_path),
                out_store=store,
                out_resolutions=out_resolutions,
                lowdim_keys=lowdim_keys,
                image_keys=rgb_keys,
                n_decoding_threads=2,
                n_encoding_threads=2,
                max_inflight_tasks=10,
                verify_read=True
            )
            
            # 验证加载的数据
            if not self._validate_replay_buffer():
                raise RuntimeError("从原始数据加载的数据无效")
            
            print(f"数据加载成功: {self.replay_buffer.n_steps} 步, {self.replay_buffer.n_episodes} episodes")
            
        except Exception as e:
            print(f"从原始数据加载失败: {e}")
            raise

    def _extract_episode_data(self, episode_idx: int) -> Dict:
        """提取指定episode的数据"""

        # 验证数据是否有效
        if self.replay_buffer is None:
            raise ValueError("ReplayBuffer未加载")
        
        if self.replay_buffer.n_episodes == 0:
            raise ValueError("ReplayBuffer中没有episode数据")
    
        episode_data = {}
        
        # 获取episode边界
        episode_starts = self.replay_buffer.episode_ends[:] - self.replay_buffer.episode_lengths[:]
        episode_ends = self.replay_buffer.episode_ends[:]
        
        if episode_idx >= len(episode_starts):
            raise ValueError(f"Episode索引超出范围: {episode_idx} (最大: {len(episode_starts)-1})")
        
        start_idx = episode_starts[episode_idx]
        end_idx = episode_ends[episode_idx]
        
        # 提取图像数据
        rgb_keys = [k for k, v in self.shape_meta['obs'].items() if v['type'] == 'rgb']
        for key in rgb_keys:
            if key in self.replay_buffer:
                episode_data[key] = self.replay_buffer[key][start_idx:end_idx]
        
        # 提取低维数据
        lowdim_keys = [k for k, v in self.shape_meta['obs'].items() if v['type'] == 'low_dim']
        for key in lowdim_keys:
            if key in self.replay_buffer:
                episode_data[key] = self.replay_buffer[key][start_idx:end_idx]
        
        # 提取动作
        if 'action' in self.replay_buffer:
            episode_data['action'] = self.replay_buffer['action'][start_idx:end_idx]
        
        # 提取时间戳
        if 'timestamp' in self.replay_buffer:
            episode_data['timestamp'] = self.replay_buffer['timestamp'][start_idx:end_idx]
        
        episode_data['episode_idx'] = episode_idx
        episode_data['start_idx'] = start_idx
        episode_data['end_idx'] = end_idx
        episode_data['length'] = end_idx - start_idx
        
        return episode_data
    
    def _create_info_panel(self, info_dict: Dict, width: int) -> np.ndarray:
        """创建信息面板"""
        # 计算所需高度
        lines = self._format_info_lines(info_dict)
        line_height = 25
        margin = 10
        self.info_panel_height = len(lines) * line_height + margin * 2 + 50  # 额外50像素用于进度条
        
        # 创建信息面板
        info_panel = np.zeros((self.info_panel_height, width, 3), dtype=np.uint8)
        info_panel[:] = self.colors['background']
        
        # 绘制标题
        title = f"RealPushT Dataset Visualization"
        cv2.putText(info_panel, title, (margin, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['highlight'], 2)
        
        # 绘制信息行
        y_offset = 60
        for i, (label, value) in enumerate(lines):
            text = f"{label}: {value}"
            color = self.colors['highlight'] if i == 0 else self.colors['text']
            cv2.putText(info_panel, text, (margin, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制进度条
        progress_y = self.info_panel_height - 40
        progress_width = width - 2 * margin
        
        # 进度条背景
        cv2.rectangle(info_panel, 
                     (margin, progress_y), 
                     (margin + progress_width, progress_y + 20),
                     self.colors['progress_bg'], -1)
        
        # 进度条前景
        if self.episode_data['length'] > 0:
            progress = min(1.0, self.current_step / max(1, self.episode_data['length'] - 1))
            progress_fill = int(progress_width * progress)
            cv2.rectangle(info_panel, 
                         (margin, progress_y), 
                         (margin + progress_fill, progress_y + 20),
                         self.colors['progress_bar'], -1)
        
        # 进度文本
        progress_text = f"Progress: {self.current_step}/{self.episode_data['length'] - 1}"
        cv2.putText(info_panel, progress_text, (margin, self.info_panel_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return info_panel
    
    def _format_info_lines(self, info_dict: Dict) -> List[Tuple[str, str]]:
        """格式化信息行"""
        lines = []
        
        # Episode信息
        lines.append(("Episode", f"{info_dict['episode_idx']}"))
        lines.append(("Step", f"{self.current_step} / {self.episode_data['length'] - 1}"))
        
        # 时间戳
        if 'timestamp' in info_dict:
            lines.append(("Timestamp", f"{info_dict['timestamp']:.3f}s"))
        
        # 动作
        if 'action' in info_dict:
            action_str = np.array2string(info_dict['action'], precision=3)
            lines.append(("Action", action_str))
        
        # 低维状态
        lowdim_keys = [k for k, v in self.shape_meta['obs'].items() if v['type'] == 'low_dim']
        for key in lowdim_keys:
            if key in info_dict:
                value = info_dict[key]
                # 如果是1维数组，展平显示
                if isinstance(value, np.ndarray) and value.ndim > 0 and value.size == 1:
                    value_str = f"[{value.item():.3f}]"
                else:
                    value_str = np.array2string(value, precision=3)
                lines.append((key, value_str))
        
        # 控制提示
        lines.append(("", ""))
        lines.append(("Controls", "SPACE: pause/resume, Q: quit, S: save frame"))
        lines.append(("", "←/→: prev/next frame, F/B: ±10 frames, R: reset"))
        
        return lines
    
    def _concatenate_images(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """拼接多个摄像头图像"""
        if not images:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        
        # 获取所有图像
        img_list = []
        labels = []
        
        for key in sorted(images.keys()):
            if 'camera' in key:
                img = images[key]
                # 确保图像是HWC格式
                if img.shape[0] == 3:  # CHW -> HWC
                    img = img.transpose(1, 2, 0)
                elif len(img.shape) == 2:  # 灰度图 -> RGB
                    print(f"警告: {key} 是灰度图，转换为RGB格式")
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # 在这里应用缩放！
                if self.config.scale != 1.0:
                    h, w = img.shape[:2]
                    new_size = (int(w * self.config.scale), int(h * self.config.scale))
                    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
                
                img_list.append(img)
                labels.append(key)
        
        if not img_list:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        
        # 水平拼接
        concatenated = np.concatenate(img_list, axis=1)
        
        # 关键修改：将RGB转换为BGR
        concatenated = cv2.cvtColor(concatenated, cv2.COLOR_RGB2BGR)

        # 添加标签
        img_height = concatenated.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 计算每个图像的宽度
        if len(img_list) > 0:
            img_width = concatenated.shape[1] // len(img_list)
        else:
            img_width = 0
        
        for i, label in enumerate(labels):
            # 计算文本位置
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            x = i * img_width + 10
            y = 30
            
            # 绘制背景矩形
            cv2.rectangle(concatenated, 
                        (x-5, y-text_size[1]-5), 
                        (x+text_size[0]+5, y+5), 
                        (0, 0, 0), -1)
            
            # 绘制文本
            cv2.putText(concatenated, label, (x, y), 
                    font, font_scale, (255, 255, 255), thickness)

        return concatenated
    
    def _get_current_frame_data(self) -> Dict:
        """获取当前帧的数据"""
        data = {'episode_idx': self.episode_data['episode_idx']}
        
        # 获取图像（不缩放）
        rgb_keys = [k for k, v in self.shape_meta['obs'].items() if v['type'] == 'rgb']
        for key in rgb_keys:
            if key in self.episode_data and self.current_step < len(self.episode_data[key]):
                data[key] = self.episode_data[key][self.current_step]
        
        # 获取低维数据
        lowdim_keys = [k for k, v in self.shape_meta['obs'].items() if v['type'] == 'low_dim']
        for key in lowdim_keys:
            if key in self.episode_data and self.current_step < len(self.episode_data[key]):
                data[key] = self.episode_data[key][self.current_step]
        
        # 获取动作
        if 'action' in self.episode_data and self.current_step < len(self.episode_data['action']):
            data['action'] = self.episode_data['action'][self.current_step]
        
        # 获取时间戳
        if 'timestamp' in self.episode_data and self.current_step < len(self.episode_data['timestamp']):
            data['timestamp'] = self.episode_data['timestamp'][self.current_step]
        
        return data
    
    def _save_current_frame(self, frame: np.ndarray):
        """保存当前帧为图片"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frame_ep{self.episode_data['episode_idx']}_step{self.current_step}_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"帧已保存: {filename}")
    
    def _init_video_writer(self, frame_size: Tuple[int, int]):
        """初始化视频写入器"""
        # 注意：frame_size应该已经是缩放后的尺寸
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.config.video_output_path,
            fourcc,
            self.config.fps,
            frame_size
        )
        print(f"视频保存到: {self.config.video_output_path}")
        print(f"视频尺寸: {frame_size[0]}x{frame_size[1]}, FPS: {self.config.fps}")
    
    def _handle_keyboard_input(self):
        """处理键盘输入"""
        key = cv2.waitKey(max(1, 1000 // self.config.fps)) & 0xFF
        
        if key == ord(' '):  # 空格键：暂停/继续
            self.paused = not self.paused
            print("暂停" if self.paused else "继续")
        
        elif key == ord('q'):  # Q键：退出
            self.quit_requested = True
        
        elif key == ord('s'):  # S键：保存当前帧
            self.save_current = True
        
        elif key == 81 or key == ord('a'):  # 左箭头/A键：后退1帧
            self.current_step = max(0, self.current_step - 1)
        
        elif key == 83 or key == ord('d'):  # 右箭头/D键：前进1帧
            self.current_step = min(self.episode_data['length'] - 1, self.current_step + 1)
        
        elif key == ord('z'):  # F键：快进10帧
            self.current_step = min(self.episode_data['length'] - 1, self.current_step + 10)
        
        elif key == ord('c'):  # B键：后退10帧
            self.current_step = max(0, self.current_step - 10)
        
        elif key == ord('r'):  # R键：重置
            self.current_step = self.config.start_step
        
        elif key == ord('+'):  # +键：增加FPS
            self.config.fps = min(60, self.config.fps + 1)
            print(f"FPS: {self.config.fps}")
        
        elif key == ord('-'):  # -键：减少FPS
            self.config.fps = max(1, self.config.fps - 1)
            print(f"FPS: {self.config.fps}")
        
        return key
    
    def visualize(self):
        """主可视化循环"""
        if self.replay_buffer is None:
            print("错误: 数据加载失败")
            return
        
        # 验证数据
        if not self._validate_replay_buffer():
            print("错误: 数据无效")
            return
        
        print(f"数据集信息:")
        print(f"  - 总步数: {self.replay_buffer.n_steps}")
        print(f"  - Episode数: {self.replay_buffer.n_episodes}")
        print(f"  - 数据键: {list(self.replay_buffer.data.keys())}")
        
        # 提取指定episode的数据
        try:
            self.episode_data = self._extract_episode_data(self.config.episode_idx)
        except ValueError as e:
            print(f"错误: {e}")
            return
        
        print(f"\nEpisode {self.config.episode_idx} 信息:")
        print(f"  - 长度: {self.episode_data['length']} 步")
        print(f"  - 起始索引: {self.episode_data['start_idx']}")
        print(f"  - 结束索引: {self.episode_data['end_idx']}")
        print(f"  - 可用数据键: {list(k for k in self.episode_data.keys() if k not in ['episode_idx', 'start_idx', 'end_idx', 'length'])}")
        
        # 调整起始步数
        if self.config.start_step >= self.episode_data['length']:
            print(f"警告: 起始步数 {self.config.start_step} 超出episode长度 {self.episode_data['length']}")
            self.current_step = 0
        else:
            self.current_step = self.config.start_step
        
        # 限制总帧数
        if self.config.num_frames is not None:
            max_frames = min(self.config.num_frames, self.episode_data['length'] - self.current_step)
        else:
            max_frames = self.episode_data['length'] - self.current_step
        
        # 创建显示窗口
        window_name = f"RealPushT Visualizer - Episode {self.config.episode_idx}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # 设置窗口为全屏
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # 初始化视频写入器
        if self.config.save_video:
            # 先获取一帧来确定尺寸
            test_data = self._get_current_frame_data()
            test_images = {k: v for k, v in test_data.items() if 'camera' in k}
            test_concatenated = self._concatenate_images(test_images)  # 这里会应用缩放
            
            if self.config.show_info:
                # 注意：信息面板的宽度需要与缩放后的图像宽度一致
                test_info = self._create_info_panel(test_data, test_concatenated.shape[1])
                test_frame = np.vstack([test_concatenated, test_info])
            else:
                test_frame = test_concatenated
            
            # 注意：这里不再需要额外的缩放
            # 因为concatenate_images已经处理了缩放
            self._init_video_writer((test_frame.shape[1], test_frame.shape[0]))
        
        frame_count = 0
        self.save_current = False
        
        print("\n=== 控制说明 ===")
        print("空格键: 暂停/继续")
        print("Q: 退出")
        print("S: 保存当前帧为图片")
        print("左/右箭头: 前进/后退1帧")
        print("F/B: 前进/后退10帧")
        print("R: 重置到开始")
        print("+/-: 增加/减少FPS")
        print("================\n")
        
        try:
            while not self.quit_requested and frame_count < max_frames:
                # 处理键盘输入
                key = self._handle_keyboard_input()
                
                # 如果暂停，继续等待输入
                if self.paused:
                    continue
                
                # 获取当前帧数据
                frame_data = self._get_current_frame_data()
                
                # 拼接摄像头图像（内部会处理缩放）
                images = {k: v for k, v in frame_data.items() if 'camera' in k}
                concatenated_img = self._concatenate_images(images)
                
                # 创建完整显示帧
                if self.config.show_info:
                    # 注意：信息面板的宽度需要与缩放后的图像宽度一致
                    info_panel = self._create_info_panel(frame_data, concatenated_img.shape[1])
                    display_frame = np.vstack([concatenated_img, info_panel])
                else:
                    display_frame = concatenated_img
                
                # 注意：这里不再需要缩放，因为已经在_concatenate_images中处理了
                
                # 保存当前帧
                if self.save_current:
                    self._save_current_frame(display_frame)
                    self.save_current = False
                
                # 写入视频
                if self.config.save_video and self.video_writer is not None:
                    self.video_writer.write(display_frame)
                
                # 显示
                cv2.imshow(window_name, display_frame)
                
                # 更新计数器
                frame_count += 1
                if not self.paused:
                    self.current_step += 1
                
                # 检查是否到达结尾
                if self.current_step >= self.episode_data['length']:
                    print("已到达episode末尾")
                    break
        
        except KeyboardInterrupt:
            print("程序被用户中断")
        
        finally:
            # 清理资源
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"视频已保存到: {self.config.video_output_path}")
            
            cv2.destroyAllWindows()
            print("可视化完成")

def main():
    """主函数：演示可视化工具的使用"""
    # 配置可视化参数
    config = VisualizationConfig(
        dataset_path="/home/hy/Desktop/task_padp/data/task_color3_1216_test",  # 修改为您的数据集路径
        episode_idx=10,
        start_step=0,
        # num_frames=100,  # 只显示前100帧，设为None显示全部
        num_frames=None,
        fps=10,
        scale=1.0,       # 图像缩放比例
        show_info=True,
        save_video=False,
        video_output_path="visualization_output.mp4",
        cache_enabled=True
    )
    
    # 创建并运行可视化器
    visualizer = RealPushTVisualizer(config)

    # 运行可视化
    visualizer.visualize()

if __name__ == "__main__":
    main()

