# AI LipSync System

Repository khởi tạo cho đề tài nghiên cứu sinh video đồng bộ khẩu hình từ audio
và khuôn mặt tham chiếu. Dự án hiện ở giai đoạn thiết kế: source chỉ chứa các
class và entry point rỗng để thể hiện vai trò dự kiến; chưa có crawler, model,
training, inference hoặc evaluation chạy thực tế.

Tài liệu yêu cầu gốc nằm tại
[`docs/AI-based_LipSync_System_Research_Specification_v1.0.docx`](docs/AI-based_LipSync_System_Research_Specification_v1.0.docx).

## Thiết lập môi trường phát triển (Dataset Loader)

Sử dụng Python 3.10 trở lên, ưu tiên Python 3.10 trong môi trường ảo trên Windows.
`requirements.txt` bao gồm cả thư viện tiền xử lý GRID (như PyTorch/face-alignment)
và các thư viện cho Dataset Loader, demo, unit test. Vì vậy cài đặt có thể tải nhiều
thư viện lớn; nếu chỉ cần đọc manifest thì bản thân parser chỉ sử dụng standard library.

```powershell
# Tại thư mục gốc repository
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install -e .

# Tạo .env local từ file mẫu, rồi sửa đường dẫn đến GRID dataset
Copy-Item .env.example .env

# Chạy demo: không mở media
python -m src.lipsync.datasets --demo manifest --split train
python -m src.lipsync.datasets --demo shard --split train

# Decode sample và mở frame khuôn mặt bằng OpenCV
python -m src.lipsync.datasets --demo dataset --preview frame

# Chỉ chạy unit tests (không cần GRID thật)
python -m pytest tests/datasets/test_grid_dataset.py -v
```

Trên Linux/macOS dùng `source .venv/bin/activate` và `cp .env.example .env`
thay cho hai lệnh tương ứng trên Windows. `LIPSYNC_DATA_ROOT` trong `.env` phải
trỏ đến thư mục chứa `gold/` và `silver/shards/`. Không commit file `.env`.

**Lưu ý:** `pip install -e .` giúp import package ở chế độ editable, nhưng
`pyproject.toml` hiện chưa khai báo runtime dependencies. Hãy cài
`requirements.txt` trước. Cần thiết bị âm thanh hoạt động nếu sử dụng
`--preview audio` hoặc `--preview both`.

## Phạm vi source

```text
src/
  data/                     Pipeline tiền xử lý GRID Corpus (Medallion-lite, Colab-ready)
    README.md               Tài liệu hướng dẫn chi tiết & kiến trúc data pipeline
    pipeline.py             Orchestrator điều phối toàn bộ luồng & CLI
    downloader.py           Tải dữ liệu Sheffield / Zenodo fallback, HTTP range resume
    preprocess.py           Bóc frame, S3FD face crop, chuẩn hóa audio 16kHz mono
    qc.py                   Kiểm định chất lượng utterance, lọc lỗi
    shard.py                Đóng gói Silver uncompressed TAR & upload an toàn lên Drive
    split.py                Phân chia Gold speaker-disjoint (27/3/3) & tạo manifest
    state.py                Quản lý trạng thái và cơ chế resume đa phiên
    config.py               Cấu hình hằng số, ngưỡng và đường dẫn tập trung
    exceptions.py           Các ngoại lệ nghiệp vụ chuẩn hóa
    utils.py                Tiện ích atomic JSON, SHA-256, chuỗi frame liên tiếp
  lipsync/                  Thuật toán và thí nghiệm lip-sync (Placeholder)
    __main__.py             Entry point dự kiến của ứng dụng
    config.py               Cấu hình ứng dụng và thí nghiệm
    pipeline.py             Điều phối pipeline tổng
    audio/                  Pipeline audio
    video/                  Pipeline video
    models/                 Mô hình lip-sync
    evaluation/             Đánh giá kết quả
```

> **Chi tiết về Data Pipeline:** Xem tài liệu đầy đủ tại [`src/data/README.md`](src/data/README.md).
> Module `lipsync/` hiện là skeleton phục vụ giai đoạn mô hình tiếp theo.

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
