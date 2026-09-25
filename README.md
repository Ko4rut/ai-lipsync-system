# AI LipSync System

Repository khởi tạo cho đề tài nghiên cứu sinh video đồng bộ khẩu hình từ audio
và khuôn mặt tham chiếu. Dự án hiện ở giai đoạn thiết kế: source chỉ chứa các
class và entry point rỗng để thể hiện vai trò dự kiến; chưa có crawler, model,
training, inference hoặc evaluation chạy thực tế.

Tài liệu yêu cầu gốc nằm tại
[`docs/AI-based_LipSync_System_Research_Specification_v1.0.docx`](docs/AI-based_LipSync_System_Research_Specification_v1.0.docx).

## Phạm vi source

```text
src/
  data/                     Thu thập và chuẩn bị dữ liệu nghiên cứu
    pipeline.py             Crawl, Bronze, Silver, Gold và điều phối dataset
  lipsync/                  Thuật toán và thí nghiệm lip-sync
    __main__.py             Entry point dự kiến của ứng dụng
    config.py               Cấu hình ứng dụng và thí nghiệm
    pipeline.py             Điều phối pipeline tổng
    audio/                  Pipeline audio
    video/                  Pipeline video
    models/                 Mô hình lip-sync
    evaluation/             Đánh giá kết quả
```

Các module hiện chỉ có docstring và `pass`. Chúng là placeholder để thống nhất
trách nhiệm trước khi chọn dataset, baseline và mô hình. Chỉ triển khai thêm
logic khi bước nghiên cứu tương ứng bắt đầu.

## Tổ chức dữ liệu

Dữ liệu thật nằm ngoài `src/` và không được commit vào Git:

```text
Nguồn crawl
    │
    ▼
data/bronze/    Media gốc, URL, thời điểm crawl, checksum và metadata nguồn
    │
    ▼
data/silver/    Mẫu đã kiểm tra, loại trùng/lỗi, cắt đoạn và chuẩn hóa
    │
    ▼
data/gold/      Manifest, phiên bản dataset và train/validation/test split
    │
    ▼
Lip-sync experiments
```

Bronze cần được giữ nguyên để có thể tái xử lý. Mỗi mẫu Silver phải truy ngược
được về nguồn Bronze. Gold nên ưu tiên lưu manifest tham chiếu đến Silver thay vì
sao chép toàn bộ media.

## Các vai trò dự kiến

### Data pipeline

- `DataCrawler`: thu thập media và provenance metadata.
- `BronzeStage`: lưu dữ liệu nguồn nguyên bản và checksum.
- `SilverStage`: kiểm tra, loại trùng, phân đoạn và chuẩn hóa.
- `GoldStage`: tạo manifest và dataset split có phiên bản.
- `DatasetPipeline`: điều phối các bước trên.

### Lip-sync pipeline

- `AudioPipeline`: chuẩn bị audio và đặc trưng đầu vào.
- `VideoPipeline`: chuẩn bị frame, khuôn mặt và thông tin hình học.
- `LipSyncModel`: sinh chuyển động môi đồng bộ.
- `Evaluator`: đo độ đồng bộ, chất lượng hình ảnh và hiệu năng.
- `LipSyncPipeline`: điều phối preprocessing, inference, render và evaluation.

## Cấu trúc repository

```text
data/                       Dữ liệu local, không commit
  bronze/
  silver/
  gold/
docs/                       Đặc tả và kế hoạch nghiên cứu
src/
  data/                     Planning skeleton cho data pipeline
  lipsync/
    audio/                  Audio skeleton
    video/                  Video skeleton
    models/                 Model skeleton
    evaluation/             Evaluation skeleton
    __main__.py             Application entry point
    config.py               Configuration skeleton
    pipeline.py             Top-level orchestration skeleton
pyproject.toml              Metadata Python package tối thiểu
```

## Trạng thái hiện tại

Command `lipsync` và `python -m lipsync` chỉ gọi entry point rỗng rồi kết thúc.
Cài package hoặc gọi các class planning không tạo ra dữ liệu hay video lip-sync.

Hướng triển khai dự kiến được ghi tại [docs/research-plan.md](docs/research-plan.md).
