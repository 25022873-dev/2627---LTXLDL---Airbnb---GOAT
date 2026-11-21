# Bài tập lớn cuối kỳ — Inside Airbnb (Nhóm 04, Cape Town)

**Môn học:** Lập trình xử lý dữ liệu (UET.DSE2049), học kỳ 1, năm học 2026–2027.

**Đề bài chính thức:** [Bài tập lớn Inside Airbnb](https://courses.iaidev.com/programming-for-data-processing/2627-1/projects/project_airbnb.html). Yêu cầu, thời hạn và rubric áp dụng theo thông báo mới nhất trên Canvas.

## Thông tin nhóm

- **Mã lớp:** DSE2049_4
- **Tên/mã nhóm:** Nhóm 04 — mã dữ liệu `DSE2049-C04-G04-cape-town`
- **Thành phố được phân công:** Cape Town (tiền tệ ZAR), 8 khu vực phân tích `A01`–`A08`
- **Bên đặt hàng phân tích:** TODO — nhóm điền
- **Bốn mốc dữ liệu:** `2025-09-28`, `2025-12-29`, `2026-03-30`, `2026-06-29`

| Họ tên | MSSV | GitHub username | Công việc phụ trách |
|---|---|---|---|
| Nguyễn Quốc Huy | 25022873 | 25022873-dev | Tham gia phát triển dự án, tổ chức thảo luận và cùng nhóm lựa chọn KPI, lựa chọn biểu đồ phù hợp để đánh giá hiệu quả hoạt động; phụ trách viết báo cáo (Report) và quản lý repo của nhóm. |
| Hoàng Ngọc Hưng | 25022878 | ProgrammingShell | Tham gia phát triển dự án, đảm nhận phần nhiều công việc lập trình (Coding), phát hiện và nêu các lỗi (Debugging) để cả nhóm cùng xử lý, góp phần tích hợp và hoàn thiện các chức năng hệ thống; phối hợp thảo luận KPI và biểu đồ cùng nhóm. |
| Nguyễn Tiến Lĩnh | 25022914 | TODO | Tham gia phát triển dự án, phân tích yêu cầu, đề xuất và đánh giá các giải pháp cải thiện hệ thống; tham gia tổ chức thảo luận, theo dõi các vấn đề phát sinh và cùng nhóm lựa chọn KPI, biểu đồ đánh giá. |
| Khổng Nam Khánh | 25022898 | knkknk143007 | Tham gia phát triển dự án, kiểm thử hệ thống, phát hiện, ghi nhận và đánh giá lỗi; cùng nhóm lựa chọn KPI và biểu đồ phù hợp để đánh giá hiệu quả hoạt động. |
| Lương Trường Giang | 25022840 | TODO | Tham gia phát triển dự án, xây dựng mã nguồn cho một số chức năng, đối chiếu mã nguồn với Hưng để kiểm tra tính chính xác; tham gia phân tích dữ liệu, xây dựng biểu đồ trực quan hóa KPI và diễn giải kết quả. |

Mỗi thành viên cần hiểu và giải thích được toàn bộ sản phẩm của nhóm.

## Cài đặt

**Phiên bản Python:** 3.12.0 (ghi trong `.python-version`).

```bash
git clone -b HNH-RfS https://github.com/25022873-dev/2627---LTXLDL---Airbnb---GOAT.git
cd 2627---LTXLDL---Airbnb---GOAT
python -m venv .venv
.\.venv\Scripts\Activate.ps1         # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt      # mọi thư viện đều ghim phiên bản
```

**Khoá API (chỉ cần khi phải gọi LLM):** nhà cung cấp Google Gemini, biến môi trường `GEMINI_API_KEY` (tên biến đặt ở `llm.api_key_env` trong YAML). Mã nguồn chỉ đọc biến môi trường, **không tự nạp tệp `.env`**, nên phải đặt biến trong terminal trước khi chạy:

```powershell
$env:GEMINI_API_KEY="khoá-của-bạn"     # PowerShell
```

```bash
export GEMINI_API_KEY="khoá-của-bạn"   # bash
```

Không ghi khoá thật vào tệp trong repo (`.env.example` chỉ là mẫu, `.env` đã nằm trong `.gitignore`).

Kết quả LLM của 1000 review «Mẫu LLM» đã có trong `data/processed/llm_cache.jsonl`. Khi model và prompt không đổi, pipeline dùng lại cache và **không cần khoá**. Thiếu khoá thì bước `llm` tự chuyển sang chỉ dùng cache (như `--skip-llm`); nếu cache được tạo bởi model/prompt khác config hiện tại thì in cảnh báo. Đổi model hoặc sửa prompt mà muốn gán nhãn lại thì cần khoá. Nếu không có cả khoá lẫn cache, bước `llm` dừng với thông báo "Chưa có kết quả LLM nào".

**Tệp Excel phân công không cần thiết:** ID và khu vực của nhóm đã được cache ở `data/processed/refs/`. Chỉ khi xoá thư mục này mới phải đặt tệp Excel tại đường dẫn `ids_file` trong config (tệp này không commit vì `.gitignore` loại `data/*.xlsx`).

## Chạy toàn bộ pipeline (một lệnh, gồm cả LLM)

```bash
python run_pipeline.py --config configs/cape_town.yaml
```

Lệnh này chạy lần lượt 7 bước dưới đây. Dữ liệu thô được tải về `data/raw/` (cần Internet, không commit); tệp đã có và hợp lệ thì bỏ qua. Nếu có khoá API, bước `llm` chỉ gọi API cho review chưa có kết quả hợp lệ trong cache.

| Bước | Script | Việc làm | Ra |
|---|---|---|---|
| 1 | `src/download.py` | Tải `listings`, `calendar`, `reviews` của 4 mốc, kiểm tra gzip và cột khoá, thử lại tối đa 3 lần | `data/raw/<data_root>/<mốc>/*.csv.gz` |
| 2 | `src/qa.py` | Lọc ID của nhóm, ép kiểu, loại dòng trùng/xung đột khoá, gắn cờ (mã hoá, ngoại lai, toạ độ, mồ côi...), so với số liệu kỳ vọng | `data/processed/listings_clean.csv`, `reviews_clean.csv`, `rejected_*.csv`, `reports/qa_report.csv` |
| 3 | `src/llm.py` | Gán nhãn cảm xúc, ngôn ngữ và khía cạnh cho 1000 review «Mẫu LLM» bằng Gemini (theo lô, có cache, chạy dở chạy tiếp được) | `llm_cache.jsonl`, `llm_results.csv`, `llm_usage.json` |
| 4 | `src/baseline.py` | Đối chứng không dùng LLM: VADER và luật ngưỡng cố định | `baseline_results.csv` |
| 5 | `src/kpi.py` | 5 KPI theo khu vực và phường (tham số ở mục `kpi:` của YAML) | `kpi_*.csv` |
| 6 | `src/viz.py` | Vẽ 5 hình KPI bằng tiếng Việt | `figures/*.png` |
| 7 | `src/evaluate.py` | Chấm LLM, baseline và mức sàn trên 100 nhãn tay (bootstrap CI) | `eval_metrics.csv`, `eval_confusion.csv`, `eval_errors.csv` |

`evaluate` chạy cuối cùng: nếu `data/samples/hand_labels.csv` chưa có, bước này tạo bảng mẫu (cột `label` trống) rồi dừng để nhóm điền nhãn, sau đó chạy lại. Vì `viz` chạy trước `evaluate`, 5 hình KPI vẫn được vẽ khi nhãn tay thiếu hoặc lỗi.

Tuỳ chọn:

```bash
python run_pipeline.py --config configs/cape_town.yaml --skip-llm   # không gọi API, chỉ dùng cache LLM
python src/qa.py configs/cape_town.yaml                             # mỗi bước vẫn chạy riêng được (tương tự cho kpi, viz, ...)
python src/llm.py configs/cape_town.yaml --skip-llm
```

Một bước chưa hoàn tất thì pipeline dừng ngay và in tên bước; chạy lại lệnh để tiếp tục.

## Cấu hình

Mọi tham số nằm trong `configs/cape_town.yaml`:

- `city`, `data_root`, `snapshots`, `currency`, `group_code`, `ids_file`: thành phố, nguồn dữ liệu và 4 mốc (đổi mốc thì sửa danh sách `snapshots`).
- `qa.expected`: số liệu kỳ vọng của đề (1000 listing, 5000 review, 875 entire home, 125 private room, số listing 0 review và thiếu giá theo mốc, 120–130 listing mỗi khu vực). `qa.iqr_k`: hệ số IQR cho cờ ngoại lai.
- `llm`: model (`gemini-3.1-flash-lite`), tên biến khoá, `prompt_version`, `batch_size`, `rpm`, `retries`, `max_chars`, danh sách khía cạnh, giá token dùng để ước tính chi phí (kèm ngày tra giá).
- `kpi`: `min_n` (nhóm dưới số quan sát này thì không báo cáo, hiện là 10) và `min_price_coverage` (mốc có dưới tỉ lệ này listing có giá bị loại khỏi KPI giá và doanh thu, hiện là 0.8). Mốc dùng cho KPI tiền tệ được chọn theo dữ liệu, không ghi cứng tháng trong mã.
- `baseline`: ngưỡng VADER (`thr`), ngưỡng mệnh đề rõ (`mixed_min`) và nhãn mặc định; cố định trước, không chỉnh theo nhãn tay.
- `eval`: tệp nhãn tay, số nhãn (`n_labels` = 100), số lần bootstrap (`bootstrap` = 2000, dùng cả cho CI của cảm xúc ròng).
- `seed`: 42, dùng cho mọi bước ngẫu nhiên.

## Các tệp kết quả để đối chiếu

| Nhóm | Tệp |
|---|---|
| Tham chiếu | `data/processed/refs/{listing,review,area}_DSE2049-C04-G04-cape-town.csv` |
| Dữ liệu sau QA | `data/processed/listings_clean.csv` (4000 dòng = 1000 listing × 4 mốc), `reviews_clean.csv` (5000 review), `rejected_listings.csv`, `rejected_reviews.csv` (hiện chỉ có dòng tiêu đề: không dòng nào bị loại) |
| Báo cáo QA | `reports/qa_report.csv`: quy tắc, điều kiện, lý do, hành động, số dòng, trước/sau, kỳ vọng ok/error (hiện 53 dòng kỳ vọng, tất cả `ok`) |
| LLM | `data/processed/llm_cache.jsonl`, `llm_results.csv`, `llm_usage.json` (token, tỉ lệ lỗi, ước tính chi phí cả thành phố) |
| Đối chứng | `data/processed/baseline_results.csv` |
| KPI | `data/processed/kpi_hibernation_rate.csv`, `kpi_median_price.csv`, `kpi_median_revenue.csv`, `kpi_median_occupied_nights.csv`, `kpi_net_sentiment.csv` |
| Đánh giá | `data/processed/eval_metrics.csv`, `eval_confusion.csv`, `eval_errors.csv` |
| Hình | `figures/1_ty_le_ngu_dong.png` (KPI 1), `2_gia_trung_vi.png` (KPI 2), `3_doanh_thu_trung_vi.png` (KPI 3), `4_so_dem_dat_trung_vi.png` (KPI 4), `5_cam_xuc_rong.png` (KPI 5) |
| Báo cáo | `reports/report.pdf` |

## Định nghĩa KPI và biểu đồ

Mọi KPI được tính theo `analysis_area` (A01–A08) và `neighbourhood_cleansed`; hình chỉ vẽ cấp khu vực. Nhóm có dưới `kpi.min_n` (10) quan sát thì không báo cáo. KPI 2 và 3 chỉ dùng các mốc có ít nhất `kpi.min_price_coverage` (80%) listing có giá: với dữ liệu hiện tại là 09/2025 (96%), 03/2026 (98%) và 06/2026 (98%); mốc 12/2025 có 0% listing có giá nên bị loại.

| KPI | Mốc | Công thức | Biểu đồ |
|---|---|---|---|
| 1. Tỉ lệ "ngủ đông" | cả 4 | số listing có `estimated_occupancy_l365d = 0` / số listing có giá trị occupancy | heatmap khu vực × mốc (nhạt = thấp, đậm = cao, số % trong ô) |
| 2. Giá trung vị | các mốc đủ giá | trung vị `price` khác rỗng, mọi loại phòng gộp lại (`room_type = all`) | đường nối các chấm theo mốc, mỗi khu vực một màu |
| 3. Doanh thu trung vị mỗi listing | các mốc đủ giá | trung vị `estimated_revenue_l365d`, tính cả listing doanh thu 0 | đường nối các chấm theo mốc, mỗi khu vực một màu |
| 4. Số đêm đặt trung vị | cả 4 (hình dùng 3 mốc có giá) | trung vị `estimated_occupancy_l365d` trên các listing có occupancy > 0 | thanh lệch % theo mốc: giá trung vị (cam) và số đêm đặt trung vị (xanh) của mỗi khu vực so với trung vị toàn mẫu cùng mốc; vùng xám ±10% là xấp xỉ mức chung; khu vực xếp theo doanh thu trung vị trung bình, cao nhất ở trên |
| 5. Cảm xúc ròng (LLM) | mẫu LLM | trung bình +1 (positive) / 0 (mixed) / −1 (negative) trên review có `status = ok` và không phải review tự động (`auto_generated`); CI 95% bằng cluster bootstrap theo listing (2000 lần, seed 42) | trái: điểm ước tính + khoảng tin cậy + đường toàn mẫu; phải: cơ cấu negative/mixed/positive (mỗi thanh = 100%) |

Cảm xúc ròng mỗi khu vực đều cao (0,79–0,98 ở cấp khu vực, từ A07 đến A01), vì 90% review trong mẫu LLM được gán positive (900 positive, 76 mixed, 24 negative). Các khoảng tin cậy của nhiều khu vực chồng lên nhau nên không nên xếp hạng chặt giữa các khu vực.

## Hạn chế đã biết (cần nêu trong báo cáo)

- **Giá thiếu ở mốc 2025-12-29:** `price` và `estimated_revenue_l365d` trống ở cả 1000 listing, nên quy tắc `kpi.min_price_coverage` tự loại mốc này và hai KPI tiền tệ chỉ dùng 3 mốc.
- **Calendar chưa được dùng:** `download.py` tải và kiểm tra tệp `calendar.csv.gz` (gzip hợp lệ, đủ cột khoá), nhưng `qa.py` và các KPI không đọc nó. KPI lấp đầy dùng `estimated_occupancy_l365d`, vốn là ước tính của Inside Airbnb dựa trên số review, nên listing mới mở trong năm cũng có thể có occupancy 0 và bị tính vào "ngủ đông".
- **Mùa vụ:** `price` là giá tại thời điểm chụp; mốc 03 (cuối hè ở Cape Town) có trung vị giá toàn mẫu cao hơn (1.908 ZAR so với 1.435 ở mốc 09 và 1.684 ở mốc 06), không nên đọc là xu hướng tăng thuần.
- **Số đêm là cửa sổ 365 ngày trước mỗi mốc,** nên các mốc chồng lên nhau và không phải các lần đo độc lập.
- **Đánh giá LLM:** nhãn tay lệch mạnh (97 positive, 2 negative, 1 mixed). Accuracy: LLM 0,90, baseline VADER 0,95, mức sàn (luôn đoán positive) 0,97. LLM gán "mixed" cho 10 review mà người gán coi là positive; baseline làm vậy với 4 review. Macro-F1 của LLM 0,704 (CI 95% 0,316–0,806) so với baseline 0,658 (0,321–0,665); chênh lệch LLM − baseline có CI 95% (−0,021; +0,154) chứa 0 nên chưa kết luận được bên nào tốt hơn. Macro-F1 dựa vào 3 review không phải positive nên khoảng tin cậy rất rộng. Cohen's kappa so với nhãn tay: LLM 0,35, baseline 0,43; giữa LLM và baseline trên cả 1000 review: độ đồng thuận 0,904, kappa 0,327. Không chỉnh prompt dựa trên chính 100 nhãn này (sẽ làm kết quả lạc quan giả).
- **Chi phí LLM:** `llm_usage.json` ước tính gán nhãn toàn bộ 760.023 review của thành phố (mốc 2026-06-29) cỡ 101,65 USD theo giá đã ghi trong config (tra ngày 2026-11-07), nên nhóm chỉ gán nhãn mẫu 1000 review.
- **Không mở tệp nhãn tay bằng Excel:** Excel làm hỏng `review_id` 19 chữ số (thành dạng `7.74432E+17`). Dùng VS Code, hoặc Excel > Data > From Text/CSV và đặt cột `review_id` là Text. `evaluate.py` kiểm tra ID và dừng với thông báo rõ nếu phát hiện lỗi này.

## Cấu trúc repo

```
run_pipeline.py          # một lệnh chạy cả pipeline
configs/cape_town.yaml   # mọi tham số
src/                     # common (dùng chung), download, qa, llm, baseline, kpi, viz, evaluate
data/raw/                # dữ liệu thô tải về (không commit)
data/processed/          # dữ liệu sau QA, kết quả LLM, KPI, đánh giá, refs/
data/samples/            # nhãn tay (hand_labels.csv)
reports/                 # qa_report.csv và report.pdf
figures/                 # 5 hình KPI cho báo cáo
AI_USAGE.md              # minh bạch việc dùng AI
```

## Nộp bài

Sau khi hoàn tất sản phẩm, cập nhật báo cáo và đẩy commit cuối lên GitHub, tạo tag theo đề bài:

```bash
git tag final
git push origin final
git rev-parse final      # ghi SHA này vào báo cáo
```

Thời hạn, cách đăng ký và lịch vấn đáp xem tại đề bài và Canvas.
