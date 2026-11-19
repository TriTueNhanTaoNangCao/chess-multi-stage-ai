# Chess Lab — GUI Streamlit local

## Khởi chạy

Từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://localhost:8501. Server mặc định chỉ lắng nghe localhost. Bàn cờ dùng SVG của python-chess, không tải thư viện bàn cờ hoặc quân từ CDN. Giao diện dùng Streamlit >=1.67. Pygame/CLI vẫn dùng `main.py` như trước.

## Bàn đấu

- Giao diện nền đen, điểm nhấn vàng đồng; điều hướng và setup nằm phía trên. Bấm vào tên mỗi bên để mở cấu hình; seed nằm trong **Nâng cao**, FEN và giới hạn lượt nằm trong **Thiết lập**. Thay đổi cấu hình áp dụng khi bắt đầu ván mới.
- Giao diện desktop dùng toàn chiều ngang, chừa 24 px mỗi mép. Bàn cờ nằm chính giữa và chiếm phần rộng nhất, lượng giá/thống kê bên trái và biên bản SAN bên phải. Bàn cờ mở rộng theo cột giữa, không thu nhỏ theo chiều cao cửa sổ; cuộn trang để xem các điều khiển phía dưới. Trên điện thoại các phần xếp dọc và ẩn bảng phân tích phụ để ưu tiên bàn cờ. Tọa độ nằm ngoài ô cờ; đánh dấu nước vừa đi, ô hợp lệ và vua bị chiếu.
- **Người đấu AI:** chọn màu, thuật toán, depth hoặc simulations, seed; chọn checkpoint/device với bot học máy. Kéo thả quân, click quân rồi click đích, hoặc mở **Nhập nước** để nhập SAN/UCI và đầu hàng. Phong cấp chọn Hậu/Xe/Tượng/Mã. Hỗ trợ lùi nước về lượt người chơi, xoay bàn và tạo ván mới từ FEN.
- **Hai bot đấu:** cấu hình độc lập hai bên, kể cả cùng thuật toán với depth/checkpoint khác nhau. Có tạm dừng, tiếp tục, chạy một nước và khoảng nghỉ giữa các nước.
- Mỗi lần bắt đầu tạo mã trận riêng. Tạo ván mới khi ván cũ chưa xong ghi ván cũ là `aborted`, không tính thua/hòa. Đầu hàng là kết quả thắng/thua.
- Tạm dừng hoặc chờ kết thúc để tải bản ghi khi AI đang đánh. Nút tải không chạy lại ứng dụng, tránh thay đổi dữ liệu giữa lúc tải. Bàn cờ chỉ tự cập nhật khi có tác vụ AI/giải đấu cần xử lý.
- Thời gian chỉ đo `agent.act()` trên snapshot; không gồm render, chờ hiển thị hay sao chép bàn. Nodes, cutoffs, visits chỉ hiển thị khi agent cung cấp. `simulation_budget` là ngân sách cấu hình; visits gốc có thể tích lũy do tái sử dụng cây.
- Lượng giá tĩnh theo Trắng sử dụng vật chất/PST/king safety của project, đơn vị tốt. Không phải value của mạng hoặc phần trăm thắng. `last_score` trong JSON vẫn theo phía đang đi tại thời điểm tìm kiếm.

## Giải đấu

Chọn đúng hai cấu hình AI khác nhau, số ván chẵn và giới hạn ply. Hai AI đánh lần lượt từng ván, luân phiên Trắng/Đen. Bàn cờ, tiến độ và bảng xếp hạng cập nhật trong lúc chạy. Mỗi ván hoàn tất được lưu ngay, không chờ cả giải. Có thể dừng giữa giải; ván dừng mang kết quả `*`, các ván hoàn tất vẫn giữ nguyên.

Thanh setup có hai nút cấu hình Bot 1/Bot 2 và **Thể thức**. Mở Thể thức để chọn số ván và giới hạn lượt. Không có ô chọn số AI; mỗi ván luôn có một AI cầm Trắng và một AI cầm Đen. Form được thu gọn mặc định để bàn cờ luôn là nội dung chính.

Elo nội bộ khởi tạo 1200, K=32; chỉ dùng so sánh trong tập trận đang xem, không phải Elo FIDE. Cùng thuật toán/depth/checkpoint được gộp thống kê qua các seed; seed thực tế vẫn lưu trong từng trận. Giải dùng seed cấu hình cho từng ván; các cặp lặp lại cùng màu, cùng seed và cùng thế xuất phát có thể cho kết quả giống nhau.

## Lịch sử đối đầu

Dữ liệu nằm tại **`results/history.sqlite3`**, lưu sau từng nước và khi kết thúc. Sao lưu file này khi ứng dụng đã tắt. SQLite và các file WAL/SHM không được commit. Đặt biến môi trường `CHESS_HISTORY_DB` để dùng đường dẫn khác, ví dụ khi kiểm thử.

Mỗi bản ghi chứa:

- ID trận/đợt đấu, thời điểm bắt đầu/kết thúc (UTC), chế độ.
- Hai bên, cấu hình đầy đủ, đường dẫn và SHA-256 checkpoint nếu có.
- Kết quả, lý do kết thúc, lỗi bot nếu có.
- FEN khởi đầu/kết thúc, lịch sử gốc và nước UCI, chính sách nhận hòa.
- Thời gian từng nước và thống kê tìm kiếm có sẵn.

Trang **Lịch sử đối đầu** lọc theo người chơi/bot, đối thủ, kết quả và đợt thi đấu. Chọn một ván để xem nước đi, kéo thanh phát lại, xoay bàn, tải PGN/JSON. Xuất tập đang lọc thành CSV hoặc JSON.

Có thể nhập `results/tournament_results.json` cũ hoặc JSON xuất từ GUI. Dữ liệu được phát lại để kiểm tra trước khi nhập; nhập lại cùng bản ghi không nhân đôi. File cũ không có ngày thi đấu dùng thời điểm nhập và có chú thích. CLI `arena/tournament.py` cũng lưu từng ván vào `history.sqlite3` trong `--output-dir`; với mặc định `results`, GUI đọc được trực tiếp. Ba báo cáo CSV/MD/JSON CLI vẫn được ghi theo hành vi cũ; SQLite giữ các lần chạy riêng biệt.

## Checkpoint

Stage 4/5 bắt buộc checkpoint. AlphaGo có ô policy và value riêng; thiếu một mạng thì mạng đó có thể chưa huấn luyện. AlphaZero không có checkpoint dùng mạng khởi tạo. Giao diện ghi rõ tình trạng này. Checkpoint phải tồn tại và nạp đúng kiến trúc; GUI kiểm tra nghiêm ngặt state dict AlphaZero và model_class policy/value AlphaGo để tránh âm thầm dùng mạng ngẫu nhiên khi nạp nhầm file.

## Vòng đời và giới hạn

- Giữ phiên trình duyệt mở để các lượt tự động/giải đấu tiếp tục. Có thể chuyển giữa ba trang trong cùng phiên; lịch sử có nút làm mới khi đang xem giải chạy nền.
- Refresh toàn trang/đóng tab hoặc khởi động lại server có thể tạo phiên điều khiển mới. Các nước đã lưu vẫn còn để xem lại; chưa có phục hồi tác vụ đang chạy sau khi server tắt. Trận chưa xong có kết quả `*` và không được tính W/D/L.
- AI chạy trên bản sao bàn cờ trong worker nền. Mỗi phiên chỉ có một tác vụ tính nước đang chờ. Token loại kết quả cũ sau dừng/tạo ván khác.
- Tạm dừng giữ kết quả đang tính đến khi tiếp tục. Dừng ván không cưỡng chế hủy thread; lượt mới đợi tác vụ cũ trả về. Không có hard timeout hoặc đồng hồ thi đấu. Tránh depth/simulations quá cao; bot không trả về cần khởi động lại server.
- Lùi nước bị khóa khi worker còn đang tính và trong giải đấu. Không sửa kết quả ván đã kết thúc; tạo ván mới để chơi tiếp.
- Chính sách hòa theo core `claim_draw=True`. Chạm giới hạn ply được phân xử hòa với lý do riêng, không phải chiếu hòa.

## Cấu trúc và kiểm thử

`app.py` định tuyến ba trang trong `app_pages/`; `ui/board_component.py` là component Streamlit v2; `ui/live.py` bố trí bàn đấu, điều khiển và cập nhật ván; `ui/common.py` dùng chung cấu hình bot, bảng số liệu và xuất file. `ui/design.py` chứa các thành phần trình bày, `.streamlit/config.toml` định nghĩa theme. `services/game_controller.py` điều phối worker và phiên, `services/history.py` lưu SQLite/phát lại/xuất file, `services/agent_config.py` tạo bot. `arena/session.py` là logic từng lượt dùng chung với `arena/evaluator.py`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_services.py tests/test_streamlit_app.py tests/test_arena.py -q
```

Kiểm thử dịch vụ bao gồm nước sai luật, chiếu hết, phong cấp, hòa lặp, undo, kết quả worker cũ, pause/step, mở lại DB, giải đổi màu và nhập lịch sử không trùng. AppTest kiểm tra khởi tạo GUI, FEN sai và checkpoint thiếu. JavaScript bàn cờ cần kiểm tra trên trình duyệt thật.

Kiểm chứng triển khai: bộ test đầy đủ đạt **264 tests, 59 subtests**; kiểm tra kiểu bằng `ty` đạt. Các thay đổi UI cuối được kiểm tra lại bằng nhóm 11 tests mới và trình duyệt Chrome: đi quân, AI trả lời, pause/step, undo, xoay bàn, đầu hàng, phong cấp Mã, tải JSON/CSV, lịch sử và giải đổi màu. Dữ liệu thử nghiệm dùng SQLite riêng trong thư mục tạm.

Đợt đổi giao diện đen–vàng đồng: chạy lại 11 tests dịch vụ/UI và `ty`; kiểm tra Chrome cho setup dạng popover, kéo thả e2–e4 và AI trả lời, undo, xoay bàn, phong cấp Mã, tải JSON, pause/step hai bot và giải hai ván. Kiểm tra bố cục desktop, bàn cờ trên màn hình 390 px và agent API ở trang bàn đấu/lịch sử.
