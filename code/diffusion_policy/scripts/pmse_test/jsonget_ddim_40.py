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
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.10/18.28.09_diff_c_ddim_40_40_coffee_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.10/01.56.28_diff_c_ddim_40_40_coffee_preparation_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.10/10.43.47_diff_c_ddim_40_40_hammer_cleanup_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.09/14.49.34_diff_c_ddim_40_40_mug_cleanup_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.10/01.40.49_diff_c_ddim_40_40_nut_assembly_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.09/01.07.43_diff_c_ddim_40_40_pick_place_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.09/20.51.37_diff_c_ddim_40_40_square_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.08/23.08.49_diff_c_ddim_40_40_stack_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.09/14.30.26_diff_c_ddim_40_40_stack_three_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.10/19.01.10_diff_c_ddim_40_40_threading_d0/logs.json.txt"
    # input_file = "/media/disk7t/PADP1018/data/outputs/2025.11.11/01.51.29_diff_c_ddim_40_40_three_piece_assembly_d0/logs.json.txt"  # 替换为实际输入文件路径
    

    
    output_directory = "/media/disk7t/PADP1018/get_json"  # 替换为实际输出目录
    
    extract_values(input_file, output_directory)
    print("数据提取完成！")