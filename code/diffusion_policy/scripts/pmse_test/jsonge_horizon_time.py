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
    # input_file = "/media/disk7t/PADPs_Exam_Date/PADP1015_8/data/outputs/2025.10.15/10.45.42_diff_c_stack_d1/train_logs.json.txt" 

    # input_file = "/media/disk7t/PADPs_Exam_Date/PADP1015_16/data/outputs/2025.10.15/10.45.44_diff_c_stack_d1/train_logs.json.txt" 

    # input_file = "/media/disk7t/PADPs_Exam_Date/PADP1015_24/data/outputs/2025.10.15/16.17.00_diff_c_stack_d1/train_logs.json.txt" 

    # input_file = "/media/disk7t/PADPs_Exam_Date/PADP1015_32/data/outputs/2025.10.15/15.59.14_diff_c_stack_d1/train_logs.json.txt" 

    # input_file = "/media/disk7t/PADPs_Exam_Date/PADP1015_40/data/outputs/2025.10.15/16.15.28_diff_c_stack_d1/train_logs.json.txt" 

    input_file = "/media/disk7t/PADPs_Exam_Date/PADP1015_48/data/outputs/2025.10.15/20.38.45_diff_c_stack_d1/train_logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/fp/17.59.15_fp_1_40_square_d0/logs.json.txt" 

    # input_file = "/media/disk7t/PADP1018/data/outputs/fp/00.17.36_fp_1_40_stack_d0/logs.json.txt" 

    ##input_file = "/media/disk7t/PADP1018/data/outputs/padp/00.14.21_padp_c_stack_three_d0/logs.json.txt" 

    ##input_file = "/media/disk7t/PADP1018/data/outputs/padp/11.18.40_padp_c_threading_d0/logs.json.txt" 

    ##input_file = "/media/disk7t/PADP1018/data/outputs/padp/11.36.06_padp_c_three_piece_assembly_d0/logs.json.txt" 


    output_directory = "/media/disk7t/PADP1018/get_json"  # 替换为实际输出目录
    
    extract_values(input_file, output_directory)
    print("数据提取完成！")