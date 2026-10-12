# Biên bản nghiệm thu — phần Thắng

Ngày: **12/10/2026**. Môi trường: Windows, Python 3.13, chess 1.11.2, Pygame 2.6.1, NumPy 2.2.4.

## Phạm vi đã triển khai

Core board/encoder, GUI/CLI, Random + Think1/2/3 + Minimax + AlphaBeta, evaluator, round-robin logistic Elo, registry và tài liệu nhóm. Stage 3–7, model và training là skeleton có chủ sở hữu; không tính là thuật toán đã hoàn thành.

Bản đầu được lưu trong `docs/initial_code_backup.zip`, gồm games, agents, models và requirements ban đầu. Các file sai tên think1/2/3.py, minimax.py, alphabeta.py, mcts_pure.py được thay bằng tên chuẩn. Requirements bản đầu là bản freeze môi trường lớn có encoding UTF-16; đã thay bằng phụ thuộc trực tiếp.

## Kiểm thử tự động

- `python -m unittest discover tests`: 61 test.
- `python -m pytest -q`: **61 passed**, 6 subtests passed; 2 cảnh báo deprecation từ Pygame/pkg_resources.
- Luật: 20 nước đầu, chặn nước sai/null, nhập thành và qua ô bị chiếu, en-passant và en-passant bị ghim, 4 phong cấp, checkmate, stalemate, insufficient material, 50/75 nước, 3/5 lần lặp, clone giữ lịch sử.
- Agent: trả move hợp lệ, không sửa input, từ chối terminal, mate cả hai màu, Think2 tránh mate, MVV-LVA/en-passant, PST hướng đúng và đối xứng, AlphaBeta tương đương Minimax khi tắt quiescence, recapture và quiet check evasion, seed/depth.
- Encoder: HWC14, state round-trip giữ lịch sử, phát hiện metadata sai, action round-trip không trùng trên 100 vị trí ngẫu nhiên; castling, en-passant, phong cấp hai màu.
- Arena: Elo tính đúng, bảo toàn tổng rating, forfeit/exception, luân phiên màu, ply-cap riêng, tổng round-robin và 3 định dạng output.
- CLI: SAN/UCI, EOF không tự chơi, ASCII sạch, subprocess thực với nước người + bot + thoát.
- GUI SDL dummy: mapping cả hai bên, click chọn/đi, undo, 4 lựa chọn phong cấp, loại kết quả worker cũ, phím R/M/F, trạng thái mate, render và worker thực.

## Kiểm thử hồi quy sau rà soát

Thêm 18 test trong `tests/test_regressions.py`:
- Vua bắt Tốt ưu tiên hơn nước tĩnh, 30 seed cho cả hai màu.
- Reset FEN qua clone; R/F/M giữ thế cờ đã nạp.
- Null move nhập từ người dùng được yêu cầu nhập lại; bot sai kiểu/ném exception được báo lỗi.
- Encoder nhận wrapper/raw; round-trip policy/FEN reset; terminal mask và dấu value.
- Arena xử lý lỗi reset, cân bằng màu, replay giữ repetition; CLI so hai depth cùng bot.
- CLI fallback ASCII và báo tham số sai không traceback.
- GUI tự thu nhỏ cho 1920×1080, 1366×768, 1024×600; click và phong cấp sau resize, sự kiện đóng cửa sổ.

## Kiểm tra hình ảnh

Đã render Pygame bằng SDL dummy và mở kiểm tra ảnh `results/gui_preview.png`: tiếng Việt, 5 pill, quân trắng/đen, màu ô #779954 và #e9edcc, tọa độ, tiêu đề, footer. Đã sửa lỗi encoding tiếng Việt và dòng hướng dẫn sát mép bàn.

Ảnh mới chụp cửa sổ đã scale thực tế **682×628** trên SDL desktop 1024×768; cửa sổ mặc định tối đa 800×736 và có thể resize, chừa khoảng trống cho thanh tiêu đề/taskbar. Tọa độ chuột được ánh xạ ngược theo viewport.

Đây là kiểm tra render và sự kiện tự động; chưa thay thế lượt người dùng tự chơi trên cửa sổ desktop thực. Chạy `python main.py` để trình diễn.

## Giải đấu thực tế

Lệnh:

```text
python arena/tournament.py --games 2 --depth 2 --max-plies 200 --seed 42
```

Đã chạy lại benchmark sau sửa Think2. 30 ván, 1773 ply; 26 checkmate, 3 stalemate, 1 threefold repetition. Không forfeit, không hòa do max_plies. Tổng thời gian tính nước khoảng 196.0 giây (không phải tổng wall-clock của quy trình kiểm thử). AlphaBeta depth 2 thắng 10/10 ván, Elo nội bộ 1331.6. Kết quả đầy đủ được lưu trong results. Kết quả W/D/L không đổi trong mẫu seed này; lỗi Vua bắt Tốt được xác minh riêng bằng regression test.

Elo là benchmark nhỏ giữa bot nội bộ; không là đánh giá sức mạnh FIDE hoặc kết quả học máy. Không có checkpoint được huấn luyện.

## Giới hạn triển khai được công bố

- Draw claim tự động khi `claim_draw=True`; không có đồng hồ/trọng tài/touch-move.
- Tensor 14 kênh phải đi kèm metadata để khôi phục đầy đủ state. Checkpoint schema và thứ tự bàn giao model nằm trong [HANDOFF.md](HANDOFF.md); loader cho các stage học máy chưa triển khai vì model vẫn là skeleton.
- Quiescence có giới hạn tài nguyên 6 ply; search cố định độ sâu.
- GUI hủy hiệu lực kết quả cũ nhưng không cưỡng chế dừng worker đang chạy.
- Arena xử lý exception/nước sai nhưng chưa có timeout cho agent treo.
- Chưa khởi tạo Git, commit, push hoặc triển khai lên dịch vụ bên ngoài.
