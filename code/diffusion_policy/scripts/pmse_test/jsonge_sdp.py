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
    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/12.10.37_sdp_1_40_coffee_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/19.41.04_sdp_1_40_coffee_preparation_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/04.47.17_sdp_1_40_hammer_cleanup_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/18.54.15_sdp_1_40_mug_cleanup_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/03.11.20_sdp_1_40_nut_assembly_d0/logs.json.txt" 

    input_file = "/media/disk7t/PADP1018/data/outputs/sdp/19.23.25_sdp_1_40_pick_place_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/12.35.56_sdp_1_40_square_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/12.40.55_sdp_1_40_stack_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/04.00.26_sdp_1_40_stack_three_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/12.20.23_sdp_1_40_threading_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/sdp/18.44.38_sdp_1_40_three_piece_assembly_d0/logs.json.txt" 


    output_directory = "/media/disk7t/PADP1018/get_json"  # 替换为实际输出目录
    
    extract_values(input_file, output_directory)
    print("数据提取完成！")