import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from collections import Counter

# Tải dataset
data = np.load(r'dataset/train50_dataset.npy', allow_pickle=True)

import numpy as np
import json
from litellm import completion
import time
from typing import List, Dict
import os

os.environ["GEMINI_API_KEY"] = "AIzaSyD6BKmewlxhOqdBIerixihw0DJ3iYDIj9g"

def analyze_instance_with_litellm(points: np.ndarray, model: str = "gemini/gemini-2.0-flash") -> Dict:
    """
    Dùng LiteLLM để phân tích cấu trúc của một instance TSP
    """
    points_list = points.tolist()

    prompt = f"""
Bạn là chuyên gia phân tích dữ liệu không gian. 
Dưới đây là tọa độ của {len(points_list)} điểm:

{json.dumps(points_list, indent=2)}

Hãy phân tích kỹ lưỡng và trả về JSON theo định dạng sau:

{{
  "description": "Mô tả ngắn bằng tiếng Việt về cấu trúc không gian (ví dụ: cụm, đều, hình tròn, đường thẳng, v.v.)",
  "type": "clustered | uniform | circular | path | grid | complex",
  "characteristics": ["đặc điểm 1", "đặc điểm 2"],
  "challenges": "Những thách thức mà heuristic có thể gặp phải khi giải instance này"
}}

Chỉ trả lời bằng JSON, không thêm bất kỳ nội dung nào khác.
"""

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Làm sạch nội dung nếu có markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return json.loads(content)

    except Exception as e:
        print(f"Error analyzing instance: {e}")
        return {
            "description": "Phân tích thất bại",
            "type": "unknown",
            "characteristics": [],
            "challenges": f"Lỗi: {str(e)}"
        }

for i in range(data.shape[0]):
    print(analyze_instance_with_litellm(data[i]))