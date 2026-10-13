import json
import os

def extract_values(input_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输出文件路径
    score_file = os.path.join(output_dir, "test_mean_scores.txt")
    time_file = os.path.join(output_dir, "inference_times.txt")
    
    with open(input_path, 'r') as f_in, \
         open(score_file, 'w') as f_score, \
         open(time_file, 'w') as f_time:
        
        for line in f_in:
            try:
                # 解析每行JSON数据
                data = json.loads(line.strip())
                
                # 提取并保存test/mean_score
                if "test/mean_score" in data:
                    f_score.write(f"{data['test/mean_score']}\n")
                
                # 提取并保存policy/avg_inference_time_ms
                if "policy/avg_inference_time_ms" in data:
                    f_time.write(f"{data['policy/avg_inference_time_ms']}\n")
                    
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line.strip()}")

if __name__ == "__main__":
    # 配置路径 (根据实际情况修改)
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.26/10.02.56_diff_c_ddim_coffee_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.26/10.03.42_diff_c_ddim_square_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.26/14.44.11_diff_c_ddim_threading_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.26/16.05.14_diff_c_ddim_stack_three_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.26/21.12.07_diff_c_ddim_stack_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.26/22.18.43_diff_c_ddim_nut_assembly_d0/logs.json.txt"  # 替换为实际输入文件路径
    ## input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.27/01.16.27_diff_c_ddim_nut_assembly_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.27/02.39.56_diff_c_ddim_hammer_cleanup_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.27/10.28.51_diff_c_ddim_coffee_preparation_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.27/10.42.23_diff_c_ddim_mug_cleanup_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.27/18.29.57_diff_c_ddim_three_piece_assembly_d0/logs.json.txt"  # 替换为实际输入文件路径
    # input_file = "/home/hy/Desktop/PADP1018/data/outputs/2025.10.28/02.12.23_diff_c_ddim_pick_place_d0/logs.json.txt"  # 替换为实际输入文件路径
    # output_directory = "/home/hy/Desktop/PADP1018/get_json"  # 替换为实际输出目录

    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.10.31/01.57.41_padp_c_coffee_d0/train_logs.json.txt"

    input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.03/03.40.05_act_mug_cleanup_d0/logs.json.txt"
    input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.04/12.54.23_act_coffee_d0/logs.json.txt"
    input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.04/12.59.34_act_square_d0/logs.json.txt"
    
    
    output_directory = "/media/disk7t/PADP1018/get_json"  # 替换为实际输出目录
    
    extract_values(input_file, output_directory)
    print("数据提取完成！")