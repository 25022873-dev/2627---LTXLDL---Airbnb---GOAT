# AI_USAGE.md

## Công cụ đã dùng (vào việc gì)
Claude: viết code; hỗ trợ lựa chọn các cột cần lọc và gắn cờ trong qa.py dựa trên 5 KPI nhóm đề ra

## Prompt then chốt (5-10): bước nào, vào tệp nào

- prompt cho download.py:

    write me download.py that do the following:
    1. build URLs formatted as in the excel file: baseURL/snapshots/data/file
    2. build path so it stay in data/raw/data_root/snapshots/file
    3. check if files are corrupted and check key columns
    4. fetch those builts URLs and download the files
    5. If a file downloads unsuccessfully, retry up to 3 times
    6. Do not download neighbourhoods.geojson.
    7. No matter what kind of mistake the download run into, just let it retry.

- prompt cho qa.py:
    write me qa.py that do the following:

1. listings id, reviews id: Cast to integer. Rows with a null key, exact duplicates or key conflicts are removed. Only the group's IDs are read.
2. neighbourhood_cleansed, room_type: Cast to text. Empty becomes NA. Flagged if the text looks mis-encoded.
3. analysis_area: Merged in from the reference sheet. Flagged if a listing has no area.
4. price: Currency symbol and commas stripped, then cast to float. Empty, sentinel, unparseable or below 0.01 becomes NA. Zeros are counted before blanking. IQR outlier flag on log1p, per room type.
5. estimated_occupancy_l365d: Cast to float. Outside 0–365 becomes NA. Zero is kept.
6. estimated_revenue_l365d: Cast to float (or the price type). Below 0 becomes NA. Zero is kept. Optional IQR flag.
7. number_of_reviews: Cast to integer, must be ≥ 0.
8. latitude, longitude: Cast to float, range-checked. Flagged if outside the area's box.
9. reviews listing_id: Cast to integer. Flagged if it's an orphan or differs from the sheet.
10. reviews comments: Cast to text. Empty becomes NA (the LLM stage marks it empty). Encoding flag.

- prompt cho llm.py:

write me llm.py that has the following functions:

1. PROMPT, HINTS: The instructions sent to Gemini: the sentiment rules, allowed aspects with short hints, language code, and a guard that treats review text as data, never as commands. It includes examples, one of them an injection attempt.
2. AUTO: A pattern that spots auto-generated reviews (host cancellations and similar) so they can be flagged.
3. make_schema: Defines the structured output: sentiment limited to positive, mixed or negative, a language code, and aspects limited to your configured names, each positive or negative.
4. clean: Strips HTML, collapses whitespace and truncates to the character limit. baseline.py and evaluate.py reuse it, so every method sees the same text.
5. label: Sends one batch to the API. It paces calls to the rate limit, retries on rate-limit or server errors with growing waits, stops immediately on errors retrying can't fix, and re-sends any items missing or invalid in the answer. It also counts tokens.
6. main: Runs everything else. It loads the sample, builds the cache key from the model and a hash of the prompt, and reads the cache so finished reviews aren't re-sent. It also processes the remaining reviews in batches, writes each result to the cache immediately, and aborts cleanly if the API keeps failing. At the end it writes llm_results.csv and llm_usage.json with the whole-city cost estimate.

- prompt cho kpi.py:
 here the list of KPIs format as KPI: source: Formula

1. Hibernation rate: all 4 snashots: listings with occupancy = 0 / listings with non-missing occupancy, per area
2. Median price: Sep, Mar, Jun: median of non-missing price, per area * room type
3. Median revenue per listing: Sep, Mar, Jun: median of non-missing estimated_revenue_l365d, zeros included, per area
4. Median occupied nights: all 4 snapshots: median of occupancy over listings with occupancy > 0, per area
5. Net sentiment: LLM sample: mean of +1 (positive), 0 (mixed), −1 (negative) over reviews with status ok, per area
write me kpi.py according to that.

- prompt cho baseline.py:
write me baseline.py using vaderSentiment, here the functions you need and their purposes:

1. VADER, CLAUSE: VADER is the sentiment scorer, with no model training and no network. CLAUSE is a pattern that splits a review into sentences and contrasting clauses, so praise and complaints can be scored separately.
2. score: Returns VADER's overall score for a piece of text, from −1 to +1.
3. label: Turns a review into positive, mixed or negative. It says mixed if some clause is clearly positive and another clearly negative. Otherwise it compares the whole-text score to a threshold, and falls back to a default label when there's no signal. A text with no scoreable clause no longer crashes it.
4. main: Loads the same "Mẫu LLM" sample as llm.py, cleans each review with the same clean, labels it, and writes baseline_results.csv with review_id, sentiment and status. Empty text gets status empty.
keep the code as short as under 50 lines

- prompt cho evaluate.py:
    write me evaluate.py
        Functions and purposes:

        f1, macro_f1: F1 for one class, and the average over the classes that actually appear in your hand labels.
        metrics: Accuracy and macro-F1 for a group of reviews. For the full sample it also gives precision, recall and F1 per class, with the real class counts.
        kappa: Cohen's kappa, the agreement between LLM and baseline beyond chance.
        make_sheet: If the hand-label file doesn't exist, it draws a random sample from the "Mẫu LLM" reviews, fixed by the seed, writes it with an empty label column and stops so you can fill it in.

- prompt cho viz.py:
write me viz.py that give the following outputs according to the KPIs, all names and annotations must be in vietnamese:
1. Hibernation rate: heatmap; y-axis: areas, x-axis: timestamps; cells: the dormant listing percentage for one region at one particular date; numbers inside cells: exact percentage values; cell colours: represent the magnitude of the dormant listing percentage, should be like this: light blue = lower percentage, medium blue = moderate percentage, dark blue = higher percentage.
2. Median price: line chart; y-axis: median price, x-axis: timestamps; coloured lines: areas; circular markers (dots): actual median price observations at each available data collection date; legend (right): identifies each region by its colour
3. Median revenue per listing: same as median price, replace median prices with median revenue
4. Median occupied nights: small-multiple diverging horizontal bar charts; y-axis: areas, all three panels share the same row ordering, regions are sorted by average median revenue across the available dates, with the highest at the top; x-axis: percentage deviation from the overall sample median at the corresponding date, negative values extend to the left, positive values extend to the right, 0 represents the overall sample median; Each panel represents one data collection date; orange bars represent median price deviation; blue bars represent median booked nights deviation;  grey vertical band: represents the 10% range around the overall sample median, values inside this band are considered approximately at the overall level
5. Net sentiment: 2 panels:
-left panel: forest plot; y-axis: areas, x-axis: net sentiment scores; blue dots: estimated net sentiment score for each area.; semi-transparent blue lines underlapping the dots are 95% confidence intervals (the uncertainty around each area's estimated score); dashed vertical line: overall sample net sentiment; numbers beside dots: exact net sentiment estimates and sample sizes
-right panel: horizontal bar chart; y-axis: areas, x-axis: percentage of reviews, from 0% to 100%; red segments: negative reviews; grey segments: mixed reviews; blue segments: positive reviews; each horizontal bar represents 100% of the reviews for one area

- prompt cho run_pipeline.py: add --skip-llm to llm.py write run_pipeline.py so everything can be ran in a command

- prompt thay đổi cách tính khoảng tin cậy trong net sentiment và macro-f1:

Sửa KPI median price: tính giá trung vị cho tất cả loại phòng gộp lại (room_type = all). Biểu đồ sử dụng kết quả tất cả loại phòng. (hiện chỉ đang tính giá trung vị cho mỗi entire home/apt)
Vấn đề hiện có của:

* cách tính khoảng tin cậy trong net sentiment: hiện không tính khoảng tin cậy cho Net Sentiment
* cách tính Macro-F1 trong từng lần bootstrap: sử dụng `np.unique(y)` ở mỗi mẫu bootstrap, khiến tập lớp được tính có thể thay đổi giữa các lần lặp. Nếu một lớp không xuất hiện trong mẫu bootstrap, lớp đó bị loại khỏi phép tính Macro-F1 thay vì tiếp tục được tính. Các lần bootstrap có thể tính Macro-F1 trên số lượng lớp khác nhau, khiến kết quả giữa các lần lặp khó so sánh trực tiếp. Khi lớp hiếm biến mất, Macro-F1 có thể tăng do lớp có F1 thấp bị loại khỏi phép tính, khiến khoảng tin cậy không phản ánh đầy đủ biến động trên toàn bộ tập lớp ban đầu.

Sửa lại cách tính khoảng tin cậy 95% (95% CI) cho KPI Net Sentiment và Macro-F1 trong source code hiện tại.
1. Net Sentiment CI — kpi.py
Viết thêm chức năng tính CI 95% cho Net Sentiment bằng Cluster Bootstrap:
1. Data: Sử dụng các review có `status = ok` và `auto_generated = False`, giữ nguyên cách tính Net Sentiment hiện tại.
2. Cluster: Gom nhóm review theo `listing_id`, tính tổng điểm cảm xúc (sum) và số lượng review (count) của từng listing.
3. Bootstrap: Thực hiện 2.000 lần, mỗi lần lấy ngẫu nhiên số listing bằng số listing ban đầu, sampling with replacement. Khi một listing được chọn, lấy toàn bộ review thuộc listing đó. Một listing có thể được chọn nhiều lần.
4. Calculation: Trong mỗi lần Bootstrap, tính lại Net Sentiment bằng tổng điểm cảm xúc chia tổng số review của các listing được chọn.
5. CI: Sau 2.000 lần, lấy percentile 2.5% và 97.5% làm cận dưới và cận trên của CI 95%.
6. Output: Bổ sung `ci_lo` và `ci_hi` vào `kpi_net_sentiment.csv`. Tính CI riêng cho từng area và giữ nguyên các cột hiện tại.
2. Macro-F1 Bootstrap — evaluate.py
Sửa cách tính Macro-F1 trong từng lần Bootstrap như sau:
1. Classes: Xác định danh sách lớp từ toàn bộ nhãn thật ban đầu bằng `np.unique(y)`. Danh sách lớp được cố định và sử dụng trong tất cả các lần Bootstrap.
2. Bootstrap: Giữ nguyên Ordinary Bootstrap hiện tại: lấy mẫu review có hoàn lại, mỗi mẫu có số review bằng tập dữ liệu ban đầu, thực hiện 2.000 lần.
3. Sampling: Sử dụng cùng chỉ số lấy mẫu cho nhãn thật và nhãn dự đoán của từng phương pháp, giữ nguyên paired sampling giữa LLM và Baseline.
4. F1: Tính TP, FP, FN và F1 riêng cho từng lớp trong mỗi lần Bootstrap.
5. Zero division: Nếu mẫu số tính F1 bằng 0, quy ước F1 của lớp đó bằng 0.
6. Macro-F1: Lấy trung bình cộng F1 của toàn bộ danh sách lớp cố định, không chỉ những lớp xuất hiện trong mẫu Bootstrap.
7. CI: Từ 2.000 giá trị Macro-F1, lấy percentile 2.5% và 97.5% làm cận dưới và cận trên của CI 95%.
8. Output: Giữ nguyên `macro_f1_ci_lo` và `macro_f1_ci_hi` trong `eval_metrics.csv`.
3. Requirements
1. Giữ random seed = 42 và số lần Bootstrap = 2.000 theo cấu hình hiện tại.
2. Giữ nguyên cách tính các KPI khác, Accuracy, Precision, Recall, Cohen's Kappa, Confusion Matrix và các chỉ số đánh giá khác.
3. Giữ nguyên cấu trúc pipeline và các file đầu ra hiện tại, chỉ bổ sung hoặc sửa những phần cần thiết.
4. Đảm bảo `viz.py` vẫn hoạt động với các file CSV sau khi thay đổi.
5. Không thay đổi dữ liệu đầu vào, quy trình xử lý dữ liệu hoặc cách gọi API LLM.
6. Kiểm tra và chạy thử code sau khi sửa để đảm bảo không có lỗi.
Gửi những file đã chỉnh sửa cho tôi, không cần gửi lại cả 1 tệp zip. Giữ code càng gọn càng tốt
## AI sai ở đâu (ví dụ thật)
1. qa.py, quy tắc lỗi mã hoá: Claude đặt `\ufffd` trong chuỗi thô (raw string). Với pandas 3 (chuỗi dùng Arrow) mã báo `ArrowInvalid` và dừng. Phát hiện khi cho Claude chạy thử trên dữ liệu giả; đã sửa thành chuỗi thường.
2. qa.py, cờ ngoại lai của doanh thu: quy tắc IQR trên log1p gắn cờ "ngoại lai" cho mọi listing có doanh thu 0 (14-16% listing trên dữ liệu thật, 46/480 trong bản thử). Số 0 là giá trị hợp lệ và là tử số của KPI "ngủ đông". Đã sửa: bỏ số 0 trước khi tính tứ phân vị.
3. evaluate.py và nhãn tay: Excel làm hỏng review_id 19 chữ số (ký hiệu khoa học). Mã cũ không phát hiện, chỉ in "thiếu/lỗi 78/100" làm số liệu trông vô lý. Nhóm đối chiếu tệp nhãn tay với tệp gốc, khôi phục ID từ tệp gốc. Mã mới kiểm tra ID và dừng với thông báo rõ.
4. Đề bài: Claude nói đề "bắt buộc" xử lý calendar trong QA. Nhóm hỏi lại "đề nói ở đâu", Claude đọc lại trang đề và thừa nhận đề chỉ yêu cầu khảo sát cấu trúc từng tệp và ghi vào phần QA.
5. kpi.py, mốc cho KPI giá: lần sửa đầu của Claude chỉ chuyển danh sách tháng 09, 03, 06 từ code sang YAML. Nhóm chỉ ra rằng như vậy vẫn là hard-code và sẽ sai khi chạy data set khác. Đã thay bằng quy tắc theo dữ liệu: mốc nào có dưới 80% listing có giá (kpi.min_price_coverage) thì bị loại khỏi KPI giá và doanh thu. Chạy thật vẫn loại đúng mốc 12/2025 (0% listing có giá).
6. run_pipeline.py, tham số --config: bản của Claude dùng đường dẫn vị trí, còn đề dùng --config. Lệnh theo đề chỉ chạy vì mã bỏ qua cờ và lấy tham số kế tiếp. --config=x.yaml bị bỏ qua mà không báo gì. Nhóm phát hiện khi đối chiếu với đề, rồi sửa bằng argparse.
7. Cờ auto_generated: mã do Claude viết có tính cờ này nhưng không nơi nào dùng. Review tự động vẫn bị tính vào KPI cảm xúc ròng. Cả bản code gốc lẫn lần rà soát đầu của Claude đều bỏ sót. Nhóm phát hiện và thêm bộ lọc trong kpi.py (và viz.py dùng chung hàm scored_reviews nên biểu đồ tự khớp).
8. viz.py, biểu đồ cảm xúc lệch với KPI: kpi.py bỏ các khu vực có dưới 10 review, còn viz.py tính lại từ dữ liệu thô và vẽ cả những khu vực đó (với 1 review thì khoảng tin cậy bị NaN). Lỗi chỉ lộ ra khi rà soát code, không lộ trong dữ liệu thật vì không khu vực nào dưới ngưỡng. Đã sửa: dùng chung ngưỡng, sau đó biểu đồ đọc điểm, n và khoảng tin cậy trực tiếp từ kpi_net_sentiment.csv.
9. evaluate.py, bootstrap Macro-F1: mã dùng np.unique(y) trên từng mẫu bootstrap nên khi lớp hiếm biến mất khỏi mẫu, lớp đó bị loại khỏi phép tính và F1 thấp của nó không còn kéo Macro-F1 xuống. Khoảng tin cậy vì thế bị thổi phồng (ở bản thử trên dữ liệu giả, cận trên của LLM giảm từ 0,377 xuống 0,272 sau khi sửa). Nhóm phát hiện khi đối chiếu cách tính; đã sửa bằng danh sách lớp cố định theo nhãn tay gốc, F1 của lớp có mẫu số 0 bằng 0.
## Nhóm kiểm chứng thế nào
- Chạy lại toàn bộ quy trình trên dữ liệu thật sau mỗi lần sửa, và đọc các tệp kết quả (qa_report.csv, kpi_*.csv, eval_*.csv) thay vì tin lời AI.
- Trước khi chọn KPI, đo % thiếu của từng cột ở cả 4 mốc và kiểm tra cột estimated_* trong tệp thô. Nhờ đó biết price và doanh thu trống 100% ở mốc 12/2025, nên KPI giá và doanh thu chỉ lấy 3 mốc.
- Cho AI chạy thử trên dữ liệu giả có cài sẵn lỗi (dòng trùng, khoá rỗng, giá "N/A", occupancy > 365, doanh thu âm, toạ độ ngoài hộp, lỗi mã hoá): bắt được lỗi 1 và 2 ở trên.
- Tự gán 100 nhãn tay (không dùng AI) rồi chấm LLM và baseline VADER, kèm khoảng tin cậy bootstrap.
- Khi AI khẳng định điều gì về đề bài, đối chiếu lại với trang đề.
- Giữ hai bản thư mục (cũ và mới), chạy cùng dữ liệu và cache, rồi so từng byte các tệp đầu ra bằng compare_outputs.py(script dùng 1 lần, chỉ để kiểm tra sự khác biệt giữa 2 phiên bản).
- Kiểm tra đường gọi API thật: xoá 3 bản ghi cuối trong cache, chạy llm.py với khoá thật, kết quả ok=1000, error=0, rồi khôi phục cache.
- Chạy python run_pipeline.py --config configs/cape_town.yaml, đọc đầu ra từng bước.
- Với phần khoảng tin cậy: AI kiểm thử trên dữ liệu giả, so cluster bootstrap với một phép tính chậm viết độc lập (cho CI gần như trùng) và kiểm tra tính lặp lại với seed 42; nhóm chạy lại pipeline trên dữ liệu thật rồi đọc kpi_net_sentiment.csv và eval_metrics.csv.