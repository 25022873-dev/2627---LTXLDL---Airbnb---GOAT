# AI_USAGE.md

## Công cụ đã dùng (vào việc gì)
Claude: viết code; hỗ trợ lựa chọn các cột cần lọc và gắn cờ trong qa.py dựa trên 5 KPI nhóm đề ra

## Prompt then chốt (5-10): bước nào, vào tệp nào

- prompt cho download.py:

    write me download.py that do the following:
        build URLs formatted as in the excel file: baseURL/snapshots/data/file
        build path so it stay in data/raw/data_root/snapshots/file
        check if files are corrupted and check key columns
        fetch those builts URLs and download the files
        If a file downloads unsuccessfully, retry up to 3 times
        Do not download neighbourhoods.geojson.
        No matter what kind of mistake the download run into, just let it retry.

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

    Hibernation rate: all 4 snashots: listings with occupancy = 0 / listings with non-missing occupancy, per area
    Median price: Sep, Mar, Jun: median of non-missing price, per area * room type
    Median revenue per listing: Sep, Mar, Jun: median of non-missing estimated_revenue_l365d, zeros included, per area
    Median occupied nights: all 4 snapshots: median of occupancy over listings with occupancy > 0, per area
    Net sentiment: LLM sample: mean of +1 (positive), 0 (mixed), −1 (negative) over reviews with status ok, per area
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
## AI sai ở đâu (ví dụ thật)
1. qa.py, quy tắc lỗi mã hoá: Claude đặt `\ufffd` trong chuỗi thô (raw string). Với pandas 3 (chuỗi dùng Arrow) mã báo `ArrowInvalid` và dừng. Phát hiện khi cho Claude chạy thử trên dữ liệu giả; đã sửa thành chuỗi thường.
2. qa.py, cờ ngoại lai của doanh thu: quy tắc IQR trên log1p gắn cờ "ngoại lai" cho mọi listing có doanh thu 0 (14-16% listing trên dữ liệu thật, 46/480 trong bản thử). Số 0 là giá trị hợp lệ và là tử số của KPI "ngủ đông". Đã sửa: bỏ số 0 trước khi tính tứ phân vị.
3. evaluate.py và nhãn tay: Excel làm hỏng review_id 19 chữ số (ký hiệu khoa học). Mã cũ không phát hiện, chỉ in "thiếu/lỗi 78/100" làm số liệu trông vô lý. Nhóm đối chiếu tệp nhãn tay với tệp gốc, khôi phục ID từ tệp gốc. Mã mới kiểm tra ID và dừng với thông báo rõ.
4. Đề bài: Claude nói đề "bắt buộc" xử lý calendar trong QA. Nhóm hỏi lại "đề nói ở đâu", Claude đọc lại trang đề và thừa nhận đề chỉ yêu cầu khảo sát cấu trúc từng tệp và ghi vào phần QA.
## Nhóm kiểm chứng thế nào
- Chạy lại toàn bộ quy trình trên dữ liệu thật sau mỗi lần sửa, và đọc các tệp kết quả (qa_report.csv, kpi_*.csv, eval_*.csv) thay vì tin lời AI.
- Trước khi chọn KPI, đo % thiếu của từng cột ở cả 4 mốc và kiểm tra cột estimated_* trong tệp thô. Nhờ đó biết price và doanh thu trống 100% ở mốc 12/2025, nên KPI giá và doanh thu chỉ lấy 3 mốc.
- Cho AI chạy thử trên dữ liệu giả có cài sẵn lỗi (dòng trùng, khoá rỗng, giá "N/A", occupancy > 365, doanh thu âm, toạ độ ngoài hộp, lỗi mã hoá): bắt được lỗi 1 và 2 ở trên.
- Tự gán 100 nhãn tay (không dùng AI) rồi chấm LLM và baseline VADER, kèm khoảng tin cậy bootstrap.
- Khi AI khẳng định điều gì về đề bài, đối chiếu lại với trang đề.