from openai import OpenAI
from pydantic import BaseModel, Field
import json

# 1. Định nghĩa cấu trúc dữ liệu bạn muốn nhận về
class ThongTinSinhVien(BaseModel):
    ten: str = Field(..., description="Tên của sinh viên")
    truong: str = Field(..., description="Tên trường học")

# 2. Lấy JSON Schema từ Pydantic model
json_schema = ThongTinSinhVien.model_json_schema()

# 3. Khởi tạo client (trỏ về vLLM local)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY", # vLLM mặc định không check key
)

# 4. Gửi request kèm tham số 'guided_json'
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system", 
            "content": "Bạn là một trợ lý AI chuyên trích xuất thông tin. Hãy trả về JSON với các trường chính xác là 'ten' và 'truong'."
        },
        {
            "role": "user", 
            "content": "tôi tên là Hưng, học ở Khoa học tự nhiên"
        }
    ],
    # ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT CỦA vLLM
    extra_body={
        "guided_json": json_schema
    },
    response_format={"type": "json_object"}
)

# 5. Kết quả trả về sẽ là chuỗi JSON chuẩn 100%
result = response.choices[0].message.content
print("Kết quả Raw:", result)

# 6. Parse lại thành object Python để dùng
data = json.loads(result)
print(f"Tên: {data.get('ten')}")
print(f"Trường: {data.get('truong')}")
