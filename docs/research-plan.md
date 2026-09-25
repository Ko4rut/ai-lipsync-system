# Kế hoạch triển khai nghiên cứu

## Câu hỏi dự kiến

Cách kết hợp đặc trưng audio–visual ảnh hưởng thế nào đến độ đồng bộ môi,
chất lượng hình ảnh và chi phí suy luận? Đây là hướng dự kiến, cần khảo sát
nghiên cứu liên quan trước khi xác nhận đóng góp mới.

## Các mốc

1. Chạy baseline Wav2Lip với source và checkpoint được ghi nhận rõ phiên bản.
   Chọn video một người, mặt rõ và audio sạch cho thử nghiệm ban đầu.
2. Chuẩn bị danh sách train/validation/test và tách người nói để tránh rò rỉ danh tính.
   Ghi nguồn dữ liệu, quy trình xử lý và điều kiện loại bỏ mẫu.
3. Tích hợp đánh giá đồng bộ, chất lượng hình ảnh và thời gian suy luận.
   PSNR/SSIM/LPIPS theo cặp chỉ dùng khi có ground truth tương ứng.
   Video thay lời thoại cần đánh giá phù hợp thay vì đối chiếu pixel với khẩu hình cũ.
4. Xây dựng thí nghiệm fusion trong cùng kiến trúc và điều kiện huấn luyện.
   So sánh concatenation với phương án attention; cố định dữ liệu, encoder,
   decoder, seed và ngân sách huấn luyện trong phạm vi có thể.
   Baseline Wav2Lip là mốc tham chiếu ngoài, không tự nó tạo ra so sánh fusion có kiểm soát.
5. Ablation, phân tích lỗi, đánh giá người xem và báo cáo giới hạn.

## Điều kiện cần chốt trước khi train

- GPU/VRAM và thời gian tính toán khả dụng.
- Bộ dữ liệu và quy mô được phép sử dụng.
- Video nguồn hay ảnh tĩnh là phạm vi chính của thực nghiệm.
- Hỗ trợ tiếng Việt là yêu cầu chính hay tập kiểm tra mở rộng.

## Nguyên tắc báo cáo

Không điền metric khi chưa có phép đo. Ghi cấu hình, phiên bản code/dependency,
checkpoint, seed và danh sách mẫu của mỗi thí nghiệm. Báo cáo riêng thời gian
tiền xử lý, suy luận và xuất video nếu đánh giá hiệu năng toàn pipeline.
