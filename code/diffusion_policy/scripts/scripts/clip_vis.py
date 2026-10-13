#!/usr/bin/env python3
"""
CLIP嵌入缓存查看工具 - 绝对安全只读版本
此脚本只读取文件，不写入任何内容
使用方法: python safe_view_clip_cache.py [--cache-dir CACHE_DIR]
"""

import os
import torch
import json
import argparse
from pathlib import Path
import warnings

def safe_load_cache_file(file_path):
    """安全加载缓存文件，只读"""
    try:
        # 只读模式加载
        embeddings = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # 尝试加载元数据（如果存在）
        meta_path = file_path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'stage_values': [], 'stage_text_prompts': {}}
            
        return embeddings, metadata
    except Exception as e:
        print(f"⚠️  加载文件失败 {file_path}: {e}")
        return None, None

def safe_analyze_embeddings(embeddings):
    """安全分析嵌入向量，不修改"""
    if embeddings is None:
        return {}
    
    # 创建副本进行分析，不修改原始
    embeddings_copy = embeddings.clone().detach()
    
    results = {}
    results['shape'] = embeddings_copy.shape
    results['dtype'] = str(embeddings_copy.dtype)
    results['device'] = str(embeddings_copy.device)
    
    # 统计信息
    with torch.no_grad():
        results['min'] = embeddings_copy.min().item()
        results['max'] = embeddings_copy.max().item()
        results['mean'] = embeddings_copy.mean().item()
        results['std'] = embeddings_copy.std().item()
        
        # 检查NaN/Inf
        results['has_nan'] = torch.isnan(embeddings_copy).any().item()
        results['has_inf'] = torch.isinf(embeddings_copy).any().item()
    
    return results

def safe_compute_similarity(embeddings):
    """安全计算相似度，不修改原始数据"""
    if embeddings is None or len(embeddings) < 2:
        return None
    
    with torch.no_grad():
        # 创建副本
        embeddings_copy = embeddings.clone().detach()
        
        # 归一化副本
        normalized = embeddings_copy / torch.norm(embeddings_copy, dim=1, keepdim=True)
        # 计算相似度
        similarity = torch.mm(normalized, normalized.t())
        
        return similarity.numpy()

def print_safe_info(file_path, embeddings, metadata):
    """打印安全信息，不修改任何内容"""
    print("\n" + "="*80)
    print(f"🔍 缓存文件: {file_path.name}")
    print(f"📁 路径: {file_path}")
    print(f"📊 文件大小: {file_path.stat().st_size / 1024:.1f} KB")
    print(f"⏰ 修改时间: {file_path.stat().st_mtime}")
    
    if embeddings is not None:
        print(f"\n📈 嵌入信息:")
        print(f"  📏 形状: {embeddings.shape}")
        print(f"  🎨 类型: {embeddings.dtype}")
        print(f"  💻 设备: {embeddings.device}")
        
        with torch.no_grad():
            print(f"  📊 数值范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
            print(f"  📈 均值: {embeddings.mean():.4f}, 标准差: {embeddings.std():.4f}")
            print(f"  ⚠️  包含NaN: {torch.isnan(embeddings).any().item()}")
            print(f"  ⚠️  包含Inf: {torch.isinf(embeddings).any().item()}")
    
    if metadata and 'stage_values' in metadata:
        print(f"\n📋 元数据:")
        print(f"  🎯 Stage数量: {len(metadata['stage_values'])}")
        print(f"  🔢 Stage值: {metadata['stage_values']}")
        
        if 'stage_text_prompts' in metadata:
            print(f"  📝 Stage文本提示:")
            for stage_value, prompt in metadata['stage_text_prompts'].items():
                print(f"    Stage {stage_value}: '{prompt}'")

def safe_verify_quality(embeddings, metadata):
    """安全验证质量，不修改"""
    if embeddings is None or len(embeddings) < 2:
        return
    
    print(f"\n✅ 嵌入质量验证 (只读):")
    
    with torch.no_grad():
        # 创建副本
        embeddings_copy = embeddings.clone().detach()
        
        # 计算范数
        norms = torch.norm(embeddings_copy, dim=1)
        print(f"  📏 向量范数范围: {norms.min().item():.3f} - {norms.max().item():.3f}")
        print(f"  📈 平均范数: {norms.mean().item():.3f}")
        
        # 归一化副本
        normalized = embeddings_copy / norms.unsqueeze(1)
        
        # 计算相似度
        similarity = torch.mm(normalized, normalized.t())
        
        # 检查对角线
        diag_diff = torch.abs(similarity.diag() - 1.0).max().item()
        print(f"  ✓ 对角线自相似度误差: {diag_diff:.6f} (<1e-5 优秀)")
        
        # 检查对称性
        sym_diff = torch.abs(similarity - similarity.t()).max().item()
        print(f"  ✓ 矩阵对称性误差: {sym_diff:.6f} (<1e-6 优秀)")
        
        if metadata and 'stage_values' in metadata:
            stage_values = metadata['stage_values']
            
            # 分析颜色stage
            color_stages = [s for s in stage_values if s in [0, 1, 2, 3, 4, 5]]
            if color_stages and len(color_stages) > 1:
                color_indices = [stage_values.index(s) for s in color_stages]
                
                # 计算颜色stage之间的平均相似度
                total_sim = 0
                count = 0
                for i in range(len(color_indices)):
                    for j in range(i+1, len(color_indices)):
                        total_sim += similarity[color_indices[i], color_indices[j]].item()
                        count += 1
                
                avg_color_sim = total_sim / count if count > 0 else 0
                print(f"  🎨 颜色stage平均相似度: {avg_color_sim:.3f}")
            
            # 分析stage 10
            if 10 in stage_values:
                stationary_idx = stage_values.index(10)
                if 'color_indices' in locals() and color_indices:
                    stationary_sims = [similarity[stationary_idx, i].item() for i in color_indices]
                    avg_stationary_sim = sum(stationary_sims) / len(stationary_sims)
                    print(f"  ⏸️  Stage 10与颜色stage平均相似度: {avg_stationary_sim:.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="安全查看CLIP嵌入缓存 - 只读版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python safe_view_clip_cache.py                     # 查看所有缓存
  python safe_view_clip_cache.py --dir ./my_cache    # 指定缓存目录
  python safe_view_clip_cache.py --verify            # 验证嵌入质量
  python safe_view_clip_cache.py --list-only         # 只列出文件
        """
    )
    
    parser.add_argument("--dir", "--cache-dir", default="./cache/clip_embeddings", 
                       dest="cache_dir", help="缓存目录路径")
    parser.add_argument("--list-only", action="store_true", 
                       help="只列出缓存文件，不显示详情")
    parser.add_argument("--verify", action="store_true", 
                       help="验证嵌入质量")
    parser.add_argument("--file", type=str, 
                       help="指定要查看的缓存文件")
    
    args = parser.parse_args()
    
    print("🔒 CLIP嵌入缓存查看工具 - 绝对安全只读版本")
    print("=" * 50)
    print(f"📁 缓存目录: {args.cache_dir}")
    print("⚠️  注意: 此工具是只读的，不会修改任何文件")
    print("=" * 50)
    
    # 检查目录是否存在
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return
    
    # 列出所有.pt文件
    cache_files = list(cache_dir.glob("*.pt"))
    if not cache_files:
        print(f"❌ 在 {cache_dir} 中未找到缓存文件")
        return
    
    print(f"📂 找到 {len(cache_files)} 个缓存文件:")
    for i, file_path in enumerate(cache_files, 1):
        file_size = file_path.stat().st_size / 1024
        print(f"  {i:3d}. {file_path.name} ({file_size:.1f} KB)")
    
    if args.list_only:
        return
    
    # 处理指定文件或所有文件
    files_to_check = []
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = cache_dir / args.file
        
        if file_path.exists():
            files_to_check = [file_path]
        else:
            print(f"❌ 文件不存在: {file_path}")
            return
    else:
        files_to_check = cache_files[:3]  # 默认只显示前3个
    
    print(f"\n🔍 正在分析 {len(files_to_check)} 个文件...")
    
    for i, file_path in enumerate(files_to_check, 1):
        print(f"\n📄 [{i}/{len(files_to_check)}] 分析文件...")
        
        # 加载并分析
        embeddings, metadata = safe_load_cache_file(file_path)
        
        if embeddings is None:
            print(f"❌ 无法加载文件: {file_path}")
            continue
        
        # 打印信息
        print_safe_info(file_path, embeddings, metadata)
        
        # 验证质量
        if args.verify:
            safe_verify_quality(embeddings, metadata)
        
        # 计算并显示相似度矩阵
        if embeddings is not None and len(embeddings) > 1:
            similarity = safe_compute_similarity(embeddings)
            if similarity is not None and len(similarity) <= 10:  # 只显示小矩阵
                print(f"\n🔗 相似度矩阵 (前{len(similarity)}x{len(similarity)})")
                for i in range(min(15, len(similarity))):  # 只显示前5行
                    row = [f"{sim:.3f}" for sim in similarity[i, :min(15, len(similarity))]]
                    print(f"  Stage {i}: [{', '.join(row)}]")
    
    print("\n" + "="*50)
    print("✅ 查看完成！所有操作都是只读的。")
    print("💾 您的缓存文件未被修改。")

if __name__ == "__main__":
    # 设置只读警告
    warnings.filterwarnings("ignore", message=".*write.*")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("💡 请检查文件权限和格式")