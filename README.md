# AI LipSync System

Source nền cho nghiên cứu sinh video đồng bộ khẩu hình từ audio và khuôn mặt tham chiếu.
Tài liệu gốc nằm trong `docs/AI-based_LipSync_System_Research_Specification_v1.0.docx`.

## Pipeline nghiên cứu

Source chia thành hai nhánh audio/video, sau đó căn chỉnh thời gian, fusion,
sinh khẩu hình, render và đánh giá. Xem [kiến trúc và cách nối module](docs/architecture.md).

`LipSyncPipeline` điều phối các module được truyền vào. Đã triển khai căn chỉnh
đặc trưng bằng nội suy theo timestamp và early fusion bằng concatenation.
Các encoder, temporal model, face processor, generator, renderer và evaluator
hiện là interface, cần triển khai/tích hợp mô hình cụ thể. Đây chưa phải pipeline
học sâu hoàn chỉnh hoặc training loop.

## Baseline Wav2Lip

- CLI kiểm tra FFmpeg và các đường dẫn baseline.
- Adapter gọi `inference.py` của Wav2Lip trong môi trường Python riêng.
- Kiểm tra stream audio/video bằng ffprobe trước khi chạy.
- Mỗi lần chạy có thư mục riêng, metadata, hash đầu vào, cấu hình và log.
- Dry run ghi kế hoạch, không sinh video hoặc điểm đánh giá.

Chưa có checkpoint, source Wav2Lip, huấn luyện fusion, hoặc triển khai metric SyncNet.
Adapter cần được kiểm chứng với source/checkpoint thực tế trước khi dùng kết quả nghiên cứu.

## Cài đặt

Chạy tại thư mục gốc bằng PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m lipsync --help
.\.venv\Scripts\python.exe -m lipsync doctor
```

Python >= 3.10 cho phần điều phối. FFmpeg và ffprobe phải có trên PATH.
Môi trường của Wav2Lip độc lập, dùng phiên bản Python/dependency phù hợp với source baseline.

## Chuẩn bị baseline

1. Đặt source Wav2Lip tại `external/Wav2Lip`, gồm `inference.py`.
2. Chuẩn bị môi trường và các trọng số phụ theo hướng dẫn của source baseline.
3. Đặt checkpoint tại `checkpoints/wav2lip.pth`.
4. Sửa `configs/wav2lip.json`, đặc biệt đường dẫn `python` đến interpreter của baseline.

Đường dẫn trong JSON tính từ thư mục chứa JSON; đường dẫn CLI tính từ thư mục đang chạy.
Không commit dữ liệu, trọng số hoặc môi trường Python vào Git.

```powershell
.\.venv\Scripts\python.exe -m lipsync doctor --config configs/wav2lip.json
.\.venv\Scripts\python.exe -m lipsync infer --config configs/wav2lip.json --audio data/raw/voice.wav --face data/raw/face.mp4 --dry-run
.\.venv\Scripts\python.exe -m lipsync infer --config configs/wav2lip.json --audio data/raw/voice.wav --face data/raw/face.mp4
```

Dry run vẫn cần FFprobe và media thực. Kết quả nằm trong `outputs/<run-id>/`.
Manifest có trạng thái `planned`, `running`, `completed`, hoặc `failed`.
`completed` chỉ nghĩa là sinh được video có stream hình và âm thanh, chưa chứng minh chất lượng đồng bộ.
Wav2Lip có thể dùng file tạm trong repository của nó; chạy tuần tự với cùng một bản source.

## Cấu trúc

```text
configs/                    Cấu hình thí nghiệm
src/lipsync/
  cli.py                    Lệnh doctor / infer
  config.py                 Đọc và kiểm tra cấu hình
  media.py                  Kiểm tra media bằng ffprobe
  contracts.py              Kiểu dữ liệu và kiểm tra timestamp/dimension
  pipeline.py               Điều phối pipeline nghiên cứu theo từng khối
  baseline.py               Chạy baseline bên ngoài và ghi manifest
  audio/                    Tiền xử lý, trích xuất, mô hình hóa thời gian
  video/                    Frame, xử lý mặt, đặc trưng thị giác
  alignment/                Căn chỉnh audio theo timestamp video
  fusion/                   Interface fusion và concatenation
  generation/               Interface sinh khẩu hình và render
  backends/                 Adapter mô hình có sẵn
  models/                   Nơi bổ sung mô hình nghiên cứu
  evaluation/               Nơi bổ sung metric thực tế
tests/                      Kiểm tra luồng điều phối
docs/                       Tài liệu và kế hoạch nghiên cứu
data/                       Dữ liệu local, không commit
checkpoints/                Trọng số local, không commit
external/                   Source baseline, không commit
outputs/                    Kết quả thí nghiệm, không commit
```

## Kiểm tra source

Không cần GPU hoặc dependency ML để chạy unit test:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python -m lipsync --help
```

Test dùng mock cho inference/encoder/render và dữ liệu số cho alignment/fusion;
không thay thế kiểm thử với mô hình thực. CLI `infer` hiện chạy baseline Wav2Lip;
pipeline nghiên cứu được sử dụng qua Python API khi cung cấp đủ các module.
Kế hoạch tiếp theo: [docs/research-plan.md](docs/research-plan.md).
