# Kiến trúc pipeline nghiên cứu

```text
Audio file                                Video / reference face
    |                                              |
AudioPreprocessor                         VideoPreprocessor
(decode, mono, resample)                   (decode + timestamps)
    |                                              |
AudioFeatureExtractor                     FaceProcessor
(mel / MFCC / pretrained encoder)          (detect, track, landmarks, align)
    |                                              |
AudioTemporalModel                        VisualFeatureExtractor
(pattern ngữ âm theo thời gian)            (mouth, geometry, identity)
    |                                              |
    +----------- TemporalAligner ------------------+
                        |
                   FusionModule
                        |
                   LipGenerator <--- face crops / reference
                        |
                  VideoRenderer <--- source frames / transforms / audio
                        |
                  Video + Evaluator
```

## Trách nhiệm và trạng thái

| Module | Đầu vào → đầu ra | Hiện trạng |
| --- | --- | --- |
| `audio/pipeline.py` | File → waveform → acoustic features → temporal features | Điều phối + interface từng bước |
| `video/pipeline.py` | File → frames → aligned faces → visual features | Điều phối + interface từng bước |
| `alignment/temporal.py` | Hai chuỗi đặc trưng → chuỗi chung timestamp | Nội suy tuyến tính chạy được |
| `fusion/concatenation.py` | Audio `[T, Da]`, visual `[T, Dv]` → `[T, Da+Dv]` | Concatenation chạy được |
| `generation/base.py` | Đặc trưng hợp nhất → face crops → video | Interface generator và renderer |
| `evaluation/base.py` | Video và audio → metric đo thực tế | Interface |
| `pipeline.py` | Nối tất cả khối, kiểm tra timeline và đầu ra | Điều phối chạy được với component được cung cấp |
| `baseline.py` | Gọi nguyên hệ thống Wav2Lip qua subprocess | Adapter độc lập |

Các interface dùng `Protocol`: implementation chỉ cần đáp ứng chữ ký hàm,
không bắt buộc kế thừa. Không cung cấp encoder/generator giả làm mô hình thực.

## Hợp đồng dữ liệu

- `AudioSignal`: waveform mono và sample rate.
- `FeatureSequence`: vector có cùng số chiều, mỗi vector có timestamp tính bằng giây.
- `FaceTrack`: frame nguồn, crop khuôn mặt, landmark và biến đổi từ crop về ảnh gốc.
- `VisualRepresentation`: track và đặc trưng trên cùng timeline.
- `AlignedFeatures`: audio và visual có timestamp giống nhau.
- `VideoFrames`: frame/crop và timestamp tương ứng.

VideoPreprocessor phải quy định cách xử lý ảnh tĩnh, độ dài video và frame rate.
Ảnh tĩnh cần tạo timeline đích tường minh; pipeline hiện không tự lặp ảnh/video.
FaceProcessor phải xử lý mất mặt/nhiều mặt tường minh, không âm thầm bỏ frame.
Audio feature timestamps cần phản ánh vị trí cửa sổ của encoder trên timeline chung.

Aligner mặc định báo lỗi khi timestamp video nằm ngoài phạm vi đặc trưng audio.
Chọn `boundary="hold"` mới cho phép giữ vector biên. Nội suy chỉ giải quyết
khác tốc độ lấy mẫu, không tự sửa audio/video bị lệch nguồn hoặc học phoneme–viseme.

## Nối các module

Ví dụ lắp ghép dưới đây yêu cầu cung cấp các implementation được ghi chú:

```python
from lipsync.audio import AudioPipeline
from lipsync.video import VideoPipeline
from lipsync.alignment import LinearTemporalAligner
from lipsync.fusion import ConcatenationFusion
from lipsync.pipeline import LipSyncPipeline

pipeline = LipSyncPipeline(
    audio=AudioPipeline(audio_preprocessor, audio_encoder, audio_temporal_model),
    video=VideoPipeline(video_preprocessor, face_processor, visual_encoder),
    align=LinearTemporalAligner(boundary="error"),
    fusion=ConcatenationFusion(),
    generate=lip_generator,
    render=video_renderer,
    evaluate=None,
)
result = pipeline.run(audio_path, face_path, output_path)
```

`tests/test_research_pipeline.py` có ví dụ nối toàn luồng bằng test doubles.
Ví dụ test kiểm chứng wiring, không sinh video thực.

## Bước triển khai tiếp theo

Tích hợp preprocessing và encoder thực, sau đó generator/renderer tương thích
với đầu ra fusion. Encoder và generator phải được huấn luyện cùng hợp đồng đặc trưng;
không thể nối tùy ý encoder mới vào checkpoint Wav2Lip cũ.

Biểu diễn tuple hiện tại là tham chiếu cho luồng suy luận, không giữ autograd.
Khi triển khai huấn luyện, cần biểu diễn tensor theo batch, mask/padding,
device/dtype và các module học được dưới `models/`. Cross-attention sẽ cần
giữ context audio và mask thay vì chỉ một vector đã nội suy cho mỗi frame.

CLI `infer` vẫn dùng `baseline.py`; source Wav2Lip tự xử lý các bước bên trong.
Nó không đi qua các module nghiên cứu vừa tách và không dùng để kết luận
hiệu quả của fusion mới khi chưa có thực nghiệm kiểm soát.
