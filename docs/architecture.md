# Kiến trúc dự kiến

Repository hiện chỉ mô tả ranh giới trách nhiệm. Chưa có implementation hoặc
quyết định cuối cùng về dataset, baseline và kiến trúc mô hình.

```text
Data sources
    │
    ▼
DataCrawler → BronzeStage → SilverStage → GoldStage
                                            │
                                            ▼
                                    Experiment dataset
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
               AudioPipeline                                 VideoPipeline
                     └──────────────────────┬──────────────────────┘
                                            ▼
                                      LipSyncModel
                                            ▼
                                        Evaluator
```

`src/data/` giữ placeholder cho quá trình xây dựng dataset. `src/lipsync/` chia
sẵn ranh giới `audio`, `video`, `models`, `evaluation`, `config` và pipeline tổng.
Các class chỉ có docstring và `pass`; sơ đồ trên là planning, không mô tả một hệ
thống hiện đã chạy được.

Khi bắt đầu triển khai, mỗi bước cần có đầu vào, đầu ra, điều kiện kiểm tra và
cách tái tạo kết quả rõ ràng trước khi tách thành nhiều module nhỏ hơn.
