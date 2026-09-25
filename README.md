# AI LipSync System

Dự án nghiên cứu sinh video đồng bộ khẩu hình từ audio và khuôn mặt tham chiếu.
Repository gồm hai phần liên kết với nhau:

- `src/data/`: thu thập, kiểm tra và chuẩn bị bộ dữ liệu nghiên cứu.
- `src/lipsync/`: chạy baseline, phát triển pipeline và đánh giá mô hình lip-sync.

Dữ liệu thật nằm trong `data/`, không đặt trong `src/` và không commit vào Git.
Tài liệu yêu cầu gốc nằm tại
[`docs/AI-based_LipSync_System_Research_Specification_v1.0.docx`](docs/AI-based_LipSync_System_Research_Specification_v1.0.docx).

## Luồng dữ liệu

Dự án áp dụng cách phân tầng Bronze–Silver–Gold ở mức thư mục:

```text
Nguồn crawl
    │
    ▼
data/bronze/    Dữ liệu gốc và metadata nguồn
    │
    ▼
data/silver/    Dữ liệu đã kiểm tra, làm sạch và chuẩn hóa
    │
    ▼
data/gold/      Manifest/split sẵn sàng cho một thí nghiệm cụ thể
    │
    ▼
src/lipsync/    Train, inference và evaluation
```

| Tầng | Nội dung | Nguyên tắc |
| --- | --- | --- |
| Bronze | Media tải về, URL, thời điểm crawl, checksum và metadata gốc | Giữ nguyên để có thể tái xử lý |
| Silver | Mẫu hợp lệ đã loại lỗi/trùng, cắt đoạn và chuẩn hóa định dạng | Mỗi mẫu phải truy ngược được về Bronze |
| Gold | Manifest train/validation/test và metadata của phiên bản dataset | Ưu tiên tham chiếu file Silver thay vì sao chép media |

Code chuyển đổi giữa các tầng sẽ nằm trong `src/data/`. Thư mục này hiện mới
được tạo để chuẩn bị cho crawler và data pipeline; chưa có crawler được triển khai.
Không sửa tay dữ liệu Silver hoặc Gold nếu kết quả đó có thể được tạo lại bằng code.

## Pipeline lip-sync

`src/lipsync/` hiện có hai luồng:

1. **Baseline Wav2Lip**: CLI gọi `inference.py` từ source Wav2Lip bên ngoài,
   kiểm tra media bằng FFprobe và ghi manifest/log cho từng lần chạy.
2. **Pipeline nghiên cứu dạng module**: xử lý audio và video, căn chỉnh thời gian,
   fusion, sinh khẩu hình, render và đánh giá.

Hiện đã có nội suy đặc trưng theo timestamp và early fusion bằng concatenation.
Các encoder, face processor, generator, renderer và evaluator vẫn là interface;
repository chưa có training loop hoặc mô hình học sâu hoàn chỉnh. CLI `infer` hiện
chỉ chạy baseline Wav2Lip, chưa chạy pipeline nghiên cứu dạng module.

Xem [kiến trúc pipeline](docs/architecture.md) và
[kế hoạch nghiên cứu](docs/research-plan.md).

## Cấu trúc repository

```text
configs/                    Cấu hình thí nghiệm
data/                       Dữ liệu local, không commit
  bronze/                   Dữ liệu crawl nguyên bản
  silver/                   Dữ liệu đã làm sạch và chuẩn hóa
  gold/                     Manifest và dataset split cho thí nghiệm
docs/                       Đặc tả, kiến trúc và kế hoạch nghiên cứu
src/
  data/                     Code crawl và chuẩn bị dataset (chưa triển khai)
  lipsync/
    audio/                  Xử lý và trích xuất đặc trưng audio
    video/                  Frame, khuôn mặt và đặc trưng thị giác
    alignment/              Căn chỉnh đặc trưng theo timeline
    fusion/                 Fusion interface và concatenation
    generation/             Interface sinh khẩu hình và render
    evaluation/             Interface đánh giá
    backends/               Adapter cho mô hình có sẵn
    models/                 Nơi triển khai mô hình nghiên cứu
    baseline.py             Chạy baseline và ghi manifest
    pipeline.py             Điều phối pipeline nghiên cứu
    cli.py                  Lệnh `doctor` và `infer`
tests/                      Unit test cho điều phối, alignment và fusion
checkpoints/                Trọng số local, không commit
external/                   Source mô hình bên ngoài, không commit
outputs/                    Kết quả thí nghiệm, không commit
```

## Cài đặt

Yêu cầu Python 3.10 trở lên. FFmpeg và FFprobe phải có trong `PATH`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m lipsync --help
.\.venv\Scripts\python.exe -m lipsync doctor
```

Môi trường chạy Wav2Lip nên độc lập vì phiên bản Python và dependency của
baseline có thể khác môi trường điều phối của repository.

## Chạy baseline Wav2Lip

1. Đặt source Wav2Lip tại `external/Wav2Lip`; thư mục phải chứa `inference.py`.
2. Cài dependency theo hướng dẫn của Wav2Lip trong môi trường riêng.
3. Đặt checkpoint tại `checkpoints/wav2lip.pth`.
4. Cập nhật interpreter và đường dẫn trong `configs/wav2lip.json` nếu cần.

Kiểm tra công cụ và artifact:

```powershell
.\.venv\Scripts\python.exe -m lipsync doctor --config configs/wav2lip.json
```

Lập kế hoạch chạy mà không gọi model:

```powershell
.\.venv\Scripts\python.exe -m lipsync infer `
  --config configs/wav2lip.json `
  --audio data/bronze/voice.wav `
  --face data/bronze/face.mp4 `
  --dry-run
```

Chạy inference bằng cách bỏ cờ `--dry-run`. Mỗi lần chạy tạo một thư mục trong
`outputs/<run-id>/`, gồm manifest và log. Trạng thái `completed` chỉ xác nhận
backend đã sinh video có luồng hình và âm thanh; nó chưa chứng minh chất lượng
đồng bộ khẩu hình.

## Kiểm tra source

Các unit test hiện tại không cần GPU hoặc dependency ML:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

Test sử dụng dữ liệu số và test double để kiểm tra wiring, alignment và fusion;
chúng không thay thế kiểm thử với model, checkpoint và media thực tế.

## Quy tắc lưu trữ

- Không commit media, checkpoint, source baseline hoặc output thí nghiệm.
- Ghi URL nguồn, quyền sử dụng, checksum và thời điểm crawl cho dữ liệu Bronze.
- Tách train/validation/test theo người nói hoặc nguồn video để hạn chế rò rỉ dữ liệu.
- Mỗi dataset Gold cần có manifest và phiên bản đủ để tái tạo từ Silver.
- Mỗi kết quả nghiên cứu cần ghi cấu hình, phiên bản code, checkpoint và metric thực đo.
