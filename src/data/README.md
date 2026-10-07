# GRID Data Preprocessing Pipeline (`src/data`)

Module này triển khai toàn bộ pipeline tiền xử lý dữ liệu cho **GRID Audio-Visual Speech Corpus**, chuẩn bị dữ liệu đầu vào chuẩn hóa (Silver & Gold layer) để phục vụ huấn luyện mô hình **Wav2Lip**.

Pipeline được thiết kế theo kiến trúc **Medallion-lite**, tối ưu hóa cho môi trường **Google Colab** (tính toán tạm thời trên local SSD, lưu trữ vĩnh viễn trên Google Drive) và hỗ trợ **resumable** (tiếp tục chạy an toàn sau khi ngắt kết nối runtime).

---

## 1. Kiến trúc tổng quan (Architecture)

```text
       Remote Bronze (Sheffield / Zenodo)
                       │
                       │ [1 speaker tại một thời điểm]
                       ▼
       Colab Local SSD (/content/grid_pipeline/)
        ├── downloads/   (tệp tải về .zip / .tar)
        ├── raw/         (giải nén & chuẩn hóa cấu trúc)
        ├── silver/      (crop mặt S3FD + audio 16kHz mono)
        └── package/     (đóng gói local sN.tar)
                       │
                       │ [Upload shard & metadata vĩnh viễn]
                       ▼
       Google Drive (Persistent Silver)
        ├── silver/shards/sN.tar
        └── silver/metadata/sN.json
                       │
                       │ [Tự động dọn dẹp local SSD & tiếp tục speaker kế tiếp]
                       ▼
       Google Drive (Logical Gold)
        ├── gold/train_manifest.csv   (27 speakers)
        ├── gold/val_manifest.csv     (3 speakers)
        ├── gold/test_manifest.csv    (3 speakers)
        └── gold/split_metadata.json
```

### Nguyên tắc thiết kế quan trọng
1. **Google Drive là lưu trữ vĩnh viễn, Local SSD Colab là nơi tính toán tạm thời**: Tuyệt đối không đọc/ghi và giải nén hàng chục nghìn frame JPEG lẻ trực tiếp trên Google Drive vì I/O Google Drive rất chậm và dễ bị rate limit.
2. **Đơn vị xử lý nguyên tử là 1 Speaker**: Pipeline xử lý trọn vẹn từng speaker (Tải $\rightarrow$ Tiền xử lý $\rightarrow$ QC $\rightarrow$ Đóng gói TAR $\rightarrow$ Upload $\rightarrow$ Verify $\rightarrow$ Dọn dẹp) trước khi qua speaker tiếp theo.
3. **Không đánh lại frame index**: Tên file ảnh crop lưu theo đúng chỉ số frame gốc (`0.jpg`, `1.jpg`, `3.jpg`,...). Nếu frame 2 không phát hiện được khuôn mặt, frame 3 vẫn giữ tên `3.jpg` để khớp tuyệt đối với timestamp audio của Wav2Lip.
4. **Không resize 96x96 ở Silver**: Silver lưu crop khuôn mặt nguyên bản; việc resize về 96x96 và sinh Mel spectrogram thuộc về Wav2Lip DataLoader khi train.
5. **Loại trừ `s21`**: Speaker 21 không có video nên hoàn toàn bị loại khỏi pipeline.

---

## 2. Bản đồ source code (`src/data/`)

| File | Trách nhiệm chính |
| :--- | :--- |
| `config.py` | **Cấu hình tập trung**: Định nghĩa toàn bộ hằng số (URL nguồn Sheffield/Zenodo, FPS=25, Audio=16kHz mono, Seed split=42, các ngưỡng QC, dataclass `PipelineConfig`). Không để magic number rải rác trong code. |
| `exceptions.py` | **Hệ thống ngoại lệ chuẩn hóa**: Các class lỗi kế thừa từ `GRIDPipelineError` như `DownloadError`, `ArchiveError`, `SourceLayoutError`, `PreprocessingError`, `QCError`, `ShardError`, `PersistenceError`. |
| `utils.py` | **Hàm bổ trợ an toàn**: Ghi JSON nguyên tử (`atomic_json_write` qua temp-file + rename), tính SHA-256 (`sha256_file`), thuật toán đếm chuỗi frame liên tiếp (`max_consecutive_run`), cấu hình log. |
| `state.py` | **Quản lý trạng thái & Resume**: `StateManager` theo dõi trạng thái từng speaker (`PENDING`, `DOWNLOADING`, `EXTRACTING`, `PREPROCESSING`, `QC`, `PACKAGING`, `UPLOADING`, `DONE`, `FAILED`). Lưu file `processing_state.json` nguyên tử trên Drive. Xác minh shard và metadata theo thư mục Drive hiện tại (không phụ thuộc đường dẫn tuyệt đối cũ trong state), rồi khôi phục `DONE` nếu artifact đã hoàn tất nhưng lần chạy trước bị ngắt. |
| `downloader.py` | **Tải dữ liệu thông minh**: Tải nguồn chính từ University of Sheffield; tự động chuyển sang fallback Zenodo nếu lỗi. Hỗ trợ resume HTTP Range qua file `.part`, streaming ghi đĩa, retry với exponential backoff và kiểm tra tính toàn vẹn archive (.zip, .tar). |
| `preprocess.py` | **Xử lý hình ảnh & âm thanh**: Giải nén archive, ghép cặp utterance (video `.mpg` + audio `.wav`), decode video kiểm tra FPS, bootstrap & chạy detector khuôn mặt S3FD theo batch, lưu JPEG theo frame index gốc, convert audio sang WAV 16kHz mono (signed 16-bit PCM) qua `ffmpeg`. |
| `qc.py` | **Kiểm định chất lượng (QC)**: Kiểm tra ở cấp độ từng utterance. Phân loại 14 lý do lỗi chuẩn hóa (ví dụ: `MISSING_AUDIO`, `INVALID_FPS`, `FACE_NOT_DETECTED`, `INSUFFICIENT_CONSECUTIVE_FACE_FRAMES`). Đảm bảo một mẫu hỏng không làm sập pipeline của cả speaker. |
| `shard.py` | **Đóng gói & Upload TAR**: Gom toàn bộ mẫu hợp lệ của 1 speaker thành `sN.tar` (uncompressed để tối ưu CPU & trích xuất nhanh). Upload an toàn lên Drive thông qua file tạm `.partial`, verify kích thước và checksum SHA-256 trước khi hoàn tất. |
| `split.py` | **Phân chia tập Gold (Speaker-Disjoint)**: Chia 33 speaker hợp lệ với seed cố định 42: Train (27 speakers), Val (3 speakers: `s30`, `s34`, `s9`), Test (3 speakers: `s25`, `s2`, `s8`). Tạo ra các file manifest CSV ở cấp độ từng utterance trỏ trực tiếp vào file TAR ở Silver. |
| `pipeline.py` | **Orchestrator & CLI Entry point**: Điều phối toàn bộ luồng, kiểm tra dependencies môi trường Colab, cung cấp CLI options (`--speaker`, `--all`, `--resume`, `--force`, `--limit-utterances`, `--build-gold-only`, `--dry-run`). |

---

## 3. Cấu trúc dữ liệu đầu ra trên Google Drive

Sau khi chạy xong, dữ liệu trên Google Drive sẽ có định dạng sau:

```text
<drive_root>/
├── silver/
│   ├── shards/
│   │   ├── s1.tar           <-- Uncompressed TAR chứa các utterance hợp lệ
│   │   ├── s2.tar
│   │   └── ...              (Tuyệt đối KHÔNG có s21.tar)
│   └── metadata/
│       ├── s1.json          <-- Chi tiết kết quả QC và thống kê từng utterance
│       ├── s2.json
│       └── ...
├── gold/
│   ├── train_manifest.csv   <-- Danh sách utterance cho tập Train (1 dòng / utterance)
│   ├── val_manifest.csv     <-- Danh sách utterance cho tập Val
│   ├── test_manifest.csv    <-- Danh sách utterance cho tập Test
│   └── split_metadata.json  <-- Metadata về phân chia speaker, seed và thống kê
├── state/
│   └── processing_state.json <-- Trạng thái xử lý của toàn bộ 33 speaker
└── logs/
    └── pipeline.log
```

---

## 4. Hướng dẫn sử dụng trên Google Colab

### Bước 1: Mount Drive & Chuẩn bị code
```python
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/Ko4rut/ai-lipsync-system.git /content/ai-lipsync-system
%cd /content/ai-lipsync-system
```

### Bước 2: Cài đặt Face Detection (chuẩn Wav2Lip S3FD)
```bash
!pip install face-alignment opencv-python-headless requests tqdm
```

### Bước 3: Chạy thử nghiệm Smoke Test (10 utterance của speaker s1)
Trước khi chạy hàng giờ, hãy chạy test nhanh 10 mẫu để xác nhận toàn bộ quy trình:
```bash
!python src/data/pipeline.py --speaker s1 --limit-utterances 10 --keep-workspace
```

### Bước 4: Kiểm tra kết quả QC mẫu s1
```python
import json
meta = json.loads(open(
    "/content/drive/MyDrive/2026-2027/coding/pattern_recognition/20261002_grid_dataset/silver/metadata/s1.json"
).read())
print(f"Tổng mẫu: {meta['summary']['total_candidates']}")
print(f"Hợp lệ: {meta['summary']['valid_samples']}, Không hợp lệ: {meta['summary']['invalid_samples']}")
print("Chi tiết lỗi:", meta['invalid_reason_counts'])
```

### Bước 5: Chạy đầy đủ speaker s1
```bash
!python src/data/pipeline.py --speaker s1 --force
```

### Bước 6: Chạy toàn bộ 33 speaker (hỗ trợ tiếp tục khi gián đoạn)
```bash
!python src/data/pipeline.py --all --resume
```
> **Lưu ý**: Nếu Colab bị disconnect hoặc hết timeout, chỉ cần kết nối lại và chạy lại lệnh trên. Pipeline sẽ tự động bỏ qua các speaker đã hoàn thành và tiếp tục các speaker còn dang dở.

---

## 5. Các tùy chọn dòng lệnh (CLI Reference)

```bash
# Xử lý duy nhất một speaker
python src/data/pipeline.py --speaker s2

# Xử lý toàn bộ 33 speaker
python src/data/pipeline.py --all

# Bỏ qua các speaker đã DONE (Mặc định)
python src/data/pipeline.py --all --resume

# Ép buộc xử lý lại speaker kể cả khi đã DONE
python src/data/pipeline.py --speaker s1 --force

# Giữ lại thư mục tạm trên local SSD sau khi xong (phục vụ debug)
python src/data/pipeline.py --speaker s1 --keep-workspace

# Chỉ chạy N utterance đầu tiên của mỗi speaker (phục vụ test)
python src/data/pipeline.py --speaker s1 --limit-utterances 20

# Dry-run: in các bước dự kiến mà không tải hay ghi đĩa
python src/data/pipeline.py --all --dry-run

# Chỉ tạo lại Gold manifest từ các file metadata Silver có sẵn trên Drive
python src/data/pipeline.py --build-gold-only
```

---

## 6. Chạy Unit Test ở môi trường cục bộ (Local Development)

Bộ unit test không cần tải dữ liệu thật, không yêu cầu GPU và không cần Google Drive:

```bash
# Tạo môi trường ảo và cài đặt thư viện cần thiết
python3 -m venv .venv
source .venv/bin/activate
pip install pytest requests opencv-python-headless numpy

# Chạy toàn bộ 15 test cases
pytest tests/test_grid_pipeline.py -v
```
