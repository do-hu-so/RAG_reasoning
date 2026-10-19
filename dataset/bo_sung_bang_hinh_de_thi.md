# Nội dung Bảng 1 và Hình 1–4 trong đề thi PROCON (phần thi HEXUDON)

Tài liệu bổ sung: chép lại thành chữ nội dung Bảng 1 và Hình 1–4 của file "Đề thi Procon_Tiếng Việt.pdf". Trong file PDF, bảng và các hình này chỉ ở dạng ảnh nên hệ thống không đọc được chữ trong đó.

## Bảng 1: Danh sách loại địa hình của ô (trang 2)

Bảng 1 quy định thời gian di chuyển (số bước) và lượng nhiên liệu xe tuần tra tiêu thụ theo loại địa hình. Theo đề, số bước và nhiên liệu được tính theo loại địa hình của ô mà tác nhân đang đứng khi nhận chỉ dẫn di chuyển.
- Đồng bằng (Plain, ô màu xanh lá): thời gian di chuyển 2 bước, tiêu thụ 1 nhiên liệu.
- Núi (Mountain, ô màu nâu chấm trắng): thời gian di chuyển 3 bước, tiêu thụ 2 nhiên liệu.
- Ao (Pond, ô màu xanh dương gợn sóng): không thể đi vào, không có mức tiêu thụ nhiên liệu.
- Đường bộ (Road, ô màu xám): thời gian di chuyển 1 bước khi "Thông thoáng" (Smooth), 2 bước khi "Đông đúc" (Congested), 4 bước khi "Ùn tắc" (Traffic jam); tiêu thụ 2 nhiên liệu ở cả ba tình trạng.

## Hình 1: Bản đồ (trang 2)

Hình 1 gồm hai phần. "Ví dụ bố cục bản đồ" (Example Map Layout) là một bản đồ 10x10 ô lục giác tô màu theo loại địa hình: hàng trên cùng và hàng dưới cùng là ao, phần giữa xen kẽ đồng bằng, núi, đường bộ và một vài ô ao.

"Ví dụ toạ độ bản đồ" (Example Map Coordinates) đánh số các ô từ 0 đến 99 theo từng hàng, từ trái sang phải, từ trên xuống dưới: hàng đầu là 0–9, hàng thứ hai là 10–19, ..., hàng cuối là 90–99. Như vậy số của ô = số hàng × chiều rộng bản đồ + số cột (hàng và cột đếm từ 0).

Ghi chú suy ra từ Hình 1 (đề không viết thành chữ, nên đối chiếu lại với tài liệu kỹ thuật của máy chủ thi): các ô lục giác đặt đỉnh nhọn hướng lên, các hàng chẵn (0, 2, 4, ...) lệch sang phải nửa ô so với các hàng lẻ. Khi đó ô ở hàng r, cột c có 6 ô kề: (r, c-1) và (r, c+1); nếu r chẵn thêm (r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1); nếu r lẻ thêm (r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c). Ví dụ ô 11 (hàng 1, cột 1) kề với các ô 10, 12, 0, 1, 20, 21.

## Hình 2: Điểm đến và chuỗi nhượng quyền (trang 4)

Hình 2 "Ví dụ vị trí điểm đến" (Example of Spot Placement) cho thấy các điểm đến, vẽ bằng biểu tượng cửa hàng udon, nằm trên các ô đồng bằng. "Ví dụ chuỗi nhượng quyền" (Examples of Franchise Chain) có ba chuỗi A, B, C, phân biệt bằng hoa văn rèm cửa của cửa hàng. Trong ví dụ có 4 điểm đến: 2 điểm thuộc chuỗi A, 1 điểm thuộc chuỗi B và 1 điểm thuộc chuỗi C. Các điểm cùng chuỗi cho cùng một loại udon.

## Hình 3: Loại tác nhân và vai trò (trang 5)

Có hai loại tác nhân: xe tuần tra (Patrol Car, vẽ là xe con màu xanh) và xe tiếp nhiên liệu (Refueling Car, vẽ là xe tải màu cam). Vai trò: xe tuần tra thu thập "udon" tại điểm đến; xe tiếp nhiên liệu cung cấp nhiên liệu cho xe tuần tra khi hai xe ở cùng một ô.

## Hình 4: Di chuyển của tác nhân (trang 5)

Khi đủ nhiên liệu: xe tuần tra đứng ở ô đồng bằng với nhiên liệu 6, sau 2 bước thì sang được ô đường bộ kề bên và còn nhiên liệu 5 (rời ô đồng bằng mất 2 bước và 1 nhiên liệu, đúng theo Bảng 1).

Khi không đủ nhiên liệu: xe tuần tra ở ô đồng bằng với nhiên liệu 0 thì không thể di chuyển sang ô đường bộ kề bên; đưa ra chỉ dẫn di chuyển như vậy là không hợp lệ.
