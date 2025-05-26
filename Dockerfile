## Sử dụng image Python chính thức
#FROM python:3.10-slim
#
## Thiết lập thư mục làm việc
#WORKDIR /app
#
## Sao chép requirements và cài đặt thư viện
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#
## Sao chép toàn bộ mã nguồn vào container
#COPY ./app /app
#
## Chạy chương trình
#CMD ["python", "main.py"]
