import os
import json

# 設定專案路徑
project_path = r'C:\Users\slab\Desktop\Slab Project\Route Planner'
output_file = 'Route_Planner_Full_Code.txt'

# 定義要抓取的副檔名
target_extensions = ('.py', '.ipynb')

print(f"正在掃描專案: {project_path}")

if not os.path.exists(project_path):
    print(f"找不到路徑: {project_path}，請檢查路徑是否正確。")
else:
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(project_path):
            # 排除掉不需要的資料夾（如虛擬環境、快取、Git）
            if any(part in root for part in ['.venv', '__pycache__', '.git', 'legacy']):
                continue
                
            for file in files:
                if file.endswith(target_extensions):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, project_path)
                    
                    # 寫入檔案標記與分隔線
                    f.write(f"\n\n{'='*50}\n")
                    f.write(f"FILE_PATH: {rel_path}\n")
                    f.write(f"{'='*50}\n\n")
                    
                    try:
                        if file.endswith('.ipynb'):
                            # 針對 Jupyter Notebook 提取程式碼區塊，避免雜訊太重
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as jupyter_f:
                                nb_content = json.load(jupyter_f)
                                for cell in nb_content.get('cells', []):
                                    if cell['cell_type'] == 'code':
                                        f.write("".join(cell['source']) + "\n")
                        else:
                            # 一般 .py 檔案直接讀取
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as code_f:
                                f.write(code_f.read())
                    except Exception as e:
                        f.write(f"讀取失敗: {str(e)}")

    print(f"成功！已掃描所有檔案（包含 routing_map 內容）。")
    print(f"請將生成的 {os.path.abspath(output_file)} 上傳到 AI Studio。")