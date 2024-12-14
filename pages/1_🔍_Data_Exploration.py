import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from utils import *


#################### SET SIDEBAR, PAGE TITLE


st.set_page_config(page_title="Data Exploration", page_icon="🔍")
st.title("🔍 Inside the Data:")

st.sidebar.success("Giáo Viên Hướng Dẫn: \n # KHUẤT THUỲ PHƯƠNG")
st.sidebar.success("Học Viên:\n # NGUYỄN CHẤN NAM \n # CHẾ THỊ ANH TUYỀN")
st.sidebar.success("Ngày báo cáo: \n # 16/12/2024")

################################ BIỂU ĐỒ TỔNG QUAN VỀ BÌNH LUẬN VÀ SẢN PHẨM

san_pham = pd.read_csv('data/San_pham.csv', index_col='ma_san_pham')
danh_gia= pd.read_csv('data/Danh_gia.csv', index_col=0)
khach_hang= pd.read_csv('data/Khach_hang.csv', index_col='ma_khach_hang')

# Hàm phân loại dựa trên giá trị của cột 'so_sao'
def classify_rating(star_rating):
    if star_rating <= 4:
        return 'negative'
    elif star_rating == 5:
        return 'positive'

# Áp dụng hàm vào cột 'so_sao' để tạo cột mới 'phan_loai_danh_gia'
danh_gia['phan_loai_danh_gia'] = danh_gia['so_sao'].apply(classify_rating)


danhgia_sanpham = danh_gia.merge(san_pham,on="ma_san_pham", how='left')
df=danhgia_sanpham[['ma_khach_hang','ma_san_pham','ngay_binh_luan','gio_binh_luan','noi_dung_binh_luan','phan_loai_danh_gia','so_sao','ten_san_pham','gia_ban']]

# Đánh dấu cột có bình luận
df['co_binh_luan'] = df['noi_dung_binh_luan'].notnull() & df['noi_dung_binh_luan'].str.strip().astype(bool)

# Tính số lượng bình luận tích cực, tiêu cực và không có bình luận
# Đếm số lượng sản phẩm duy nhất
total_products = san_pham.index.nunique()
total_products_eval = df['ma_san_pham'].nunique()
total_eval= df['so_sao'].count()
total_comments = df['noi_dung_binh_luan'].count()
positive_count = df[(df['phan_loai_danh_gia'] == 'positive') & (df['co_binh_luan'])].shape[0]
negative_count = df[(df['phan_loai_danh_gia'] == 'negative') & (df['co_binh_luan'])].shape[0]
no_comment_count = df[~df['co_binh_luan']].shape[0]

# Dữ liệu cho biểu đồ
categories = ['Tích cực', 'Tiêu cực', 'Không có bình luận']
values = [positive_count, negative_count, no_comment_count]
colors = sns.color_palette("pastel", len(categories))

# # Hiển thị thông tin 
st.subheader("1. Tổng quan về đánh giá và sản phẩm")

# Tạo hai cột
col1, col2 = st.columns(2)
with col1:
    st.write(f"- SL Sản phẩm: {total_products:,}")
    st.write(f'- SL Sản phẩm có đánh giá: {total_products_eval}')
    st.write(f"- SL Đánh giá: {total_eval:,}")
    st.write(f"- SL Bình luận: {total_comments:,}")


# Cột 2: Hiển thị sản phẩm không có bình luận
with col2:
    st.write(f"- SL đánh giá tích cực: {positive_count:,}")
    st.write(f"- SL đánh giá tiêu cực: {negative_count:,}")
    st.write(f"- SL có đánh giá, nhưng không bình luận: {no_comment_count:,}")


# Vẽ biểu đồ Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(categories, values, color=colors, edgecolor='black')

# Thêm số lượng trên đầu mỗi cột
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(height),
            ha='center', fontsize=12, color='black')

# Định dạng biểu đồ
ax.set_title('Phân loại đánh giá theo sản phẩm', fontsize=14)
ax.set_ylabel('Số lượng sản phẩm', fontsize=12)
ax.set_xlabel('Loại đánh giá', fontsize=12)
ax.set_ylim(0, max(values) + 1000)

# Hiển thị biểu đồ trong Streamlit
st.pyplot(fig)

################################ BIỂU ĐỒ PHÂN TÍCH SỐ LƯỢNG BÌNH LUẬN THEO THỜI GIAN

# Hiển thị thông tin 
st.subheader("2. Số lượng bình luận theo thời gian")

# Chuyển đổi cột 'ngay_binh_luan' sang kiểu datetime, loại bỏ các giá trị không hợp lệ
df['ngay_binh_luan'] = pd.to_datetime(df['ngay_binh_luan'], format='%d/%m/%Y', errors='coerce')

# Loại bỏ các dòng có giá trị NaT (Not a Time) trong cột 'ngay_binh_luan'
df_binhluan = df.dropna(subset=['ngay_binh_luan'])

# Nhóm dữ liệu theo tháng và loại đánh giá (positive/negative)
df_binhluan['thang_nam'] = df_binhluan['ngay_binh_luan'].dt.to_period('M')  # Tạo cột 'thang_nam' theo định dạng tháng-năm
df_binhluan['phan_loai_danh_gia'] = df_binhluan['phan_loai_danh_gia'].str.lower()  # Đảm bảo cột 'phan_loai_danh_gia' là chữ thường

# Tính số lượng bình luận tích cực và tiêu cực theo tháng
monthly_comments = df_binhluan.groupby(['thang_nam', 'phan_loai_danh_gia']).size().unstack(fill_value=0)

# Lấy các tháng có trong dữ liệu
available_months = sorted(monthly_comments.index.unique())

# Đảm bảo rằng index mặc định là hợp lệ (0 <= index < length of available_months)
default_start_index = 0
default_end_index = len(available_months) - 1

# Sử dụng st.columns để tạo các cột ngang cho tháng bắt đầu và tháng kết thúc
col1, col2 = st.columns(2)

# Tạo các thanh trượt để chọn tháng bắt đầu và tháng kết thúc trong cột ngang
with col1:
    # Chọn tháng bắt đầu với tháng mặc định là tháng đầu tiên (index=default_start_index)
    start_month = st.selectbox('Chọn tháng bắt đầu', available_months, index=default_start_index)

with col2:
    # Chọn tháng kết thúc với tháng mặc định là tháng cuối cùng (index=default_end_index)
    end_month = st.selectbox('Chọn tháng kết thúc', available_months, index=default_end_index)

# Lọc dữ liệu dựa trên tháng bắt đầu và tháng kết thúc
filtered_data = monthly_comments[(monthly_comments.index >= start_month) & (monthly_comments.index <= end_month)]

# Chọn các tháng cần hiển thị (ví dụ: tháng 3, 6, 9, 12)
selected_months = filtered_data.index.month.isin([3, 6, 9, 12])

# Vẽ biểu đồ bar count với phân chia tích cực và tiêu cực
fig, ax = plt.subplots(figsize=(10, 6))

# Vẽ các thanh cho bình luận tích cực
ax.bar(filtered_data.index.astype(str)[selected_months], filtered_data['positive'][selected_months], label='Tích cực', color='#4CB391', alpha=0.7)

# Vẽ các thanh cho bình luận tiêu cực
ax.bar(filtered_data.index.astype(str)[selected_months], filtered_data['negative'][selected_months], bottom=filtered_data['positive'][selected_months], label='Tiêu cực', color='red', alpha=0.5)

# Thêm tiêu đề và nhãn
ax.set_title('Số lượng bình luận tích cực và tiêu cực theo tháng', fontsize=16)
ax.set_xlabel('Tháng', fontsize=12)
ax.set_ylabel('Số lượng bình luận', fontsize=12)

# Xoay nhãn trục X để dễ đọc
ax.set_xticklabels(filtered_data.index.astype(str)[selected_months], rotation=45)

# Hiển thị legend
ax.legend(title='Loại đánh giá')

# Hiển thị biểu đồ trong Streamlit
st.pyplot(fig)

################################ BIỂU ĐỒ PHÂN PHỐI GIÁ SẢN PHẨM

# Hiển thị thông tin
st.subheader("3. Biểu đồ phân phối giá sản phẩm")

# Vẽ biểu đồ phân phối giá sản phẩm
plt.figure(figsize=(10, 6))
sns.histplot(df['gia_ban'], kde=True, bins=30, color='#4CB391', alpha=0.7)

# Thêm tiêu đề và các nhãn
plt.title('Phân phối giá sản phẩm', fontsize=14)
plt.xlabel('Giá sản phẩm', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)

# Hiển thị biểu đồ trong Streamlit
st.pyplot(plt)

############## BIỂU ĐỒ ĐÁNH GIÁ THEO NHÓM GIÁ SẢN PHẨM

# Chia nhóm giá sản phẩm dựa trên cột gia_ban
bins = [0, 100000, 500000, float('inf')]  # Cài đặt giá trị theo nhóm giá, ví dụ giá thấp, trung bình, cao
labels = ['Giá thấp', 'Giá trung bình', 'Giá cao']
df['gia_nhom'] = pd.cut(df['gia_ban'], bins=bins, labels=labels, right=False)

# Tính toán tỉ lệ đánh giá tích cực/tiêu cực cho từng sản phẩm
sentiment_distribution = df.groupby(['ma_san_pham', 'phan_loai_danh_gia']).size().unstack(fill_value=0)
sentiment_distribution['positive_ratio'] = sentiment_distribution['positive'] / sentiment_distribution.sum(axis=1)
sentiment_distribution['negative_ratio'] = sentiment_distribution['negative'] / sentiment_distribution.sum(axis=1)

# Thêm thông tin nhóm giá vào dữ liệu sentiment_distribution
df_sentiment = df.groupby(['ma_san_pham', 'gia_nhom', 'phan_loai_danh_gia']).size().unstack(fill_value=0)

# Tính số lượng đánh giá tích cực và tiêu cực cho từng nhóm giá
sentiment_by_group = df_sentiment.groupby('gia_nhom')[['positive', 'negative']].sum()

# Chuyển dataframe sentiment_by_group thành dạng dài để seaborn vẽ biểu đồ dễ dàng
sentiment_by_group_reset = sentiment_by_group.reset_index()
sentiment_by_group_melted = sentiment_by_group_reset.melt(id_vars='gia_nhom', value_vars=['positive', 'negative'], 
                                                         var_name='Loại đánh giá', value_name='Số lượng')

# Hàm vẽ biểu đồ
def draw_sentiment_chart():

    # Tạo biểu đồ Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=sentiment_by_group_melted, 
        x='gia_nhom', 
        y='Số lượng', 
        hue='Loại đánh giá', 
        dodge=True, 
        palette='Set2'
    )

    # Thêm tiêu đề và nhãn
    plt.title('Tỉ lệ đánh giá tích cực và tiêu cực cho từng nhóm giá sản phẩm', fontsize=14)
    plt.xlabel('Nhóm giá sản phẩm', fontsize=12)
    plt.ylabel('Số lượng đánh giá', fontsize=12)
    plt.legend(title='Loại đánh giá')

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)

st.subheader("4. Tỉ lệ đánh giá theo nhóm giá sản phẩm")
draw_sentiment_chart()

############## SẢN PHẨM CÓ ĐÁNH GIÁ CAO NHẤT VÀ THẤP NHẤT

# Hiển thị thông tin 
st.subheader("5. Sản phẩm có đánh giá thấp nhất và cao nhất")

# Tính điểm đánh giá trung bình cho mỗi sản phẩm
# Giả sử cột 'danh_gia' là điểm đánh giá và cột 'san_pham' là tên sản phẩm
average_ratings = df.groupby('ma_san_pham')[['so_sao']].mean().reset_index()

# Sắp xếp các sản phẩm theo điểm đánh giá từ thấp đến cao
sorted_ratings = average_ratings.sort_values(by='so_sao')

# Data chứ link hình sản phẩm
image_df = pd.read_csv('files_nam/San_pham_Link_Image_Brand.csv')

# Kết hợp dữ liệu hình ảnh và thông tin sản phẩm
combined_df = pd.merge(average_ratings, image_df, on="ma_san_pham", how="inner")
combined_df = pd.merge(combined_df, san_pham, on="ma_san_pham", how="inner")
         
# Hàm hiển thị hình ảnh theo số lượng cố định (2 cột x 3 dòng)
def display_images(df, num_images=6):
    cols = st.columns(2)  # Chia thành 2 cột
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:  # Chia sản phẩm vào từng cột
            # Hiển thị hình ảnh sản phẩm
            st.image(row['hinh_anh'], width=300)

            # Hiển thị tên sản phẩm dưới dạng hyperlink
            st.markdown(
                f"<h5 style='text-align: center; margin: 5px;'>"
                f"<a href='{row['chi_tiet']}' target='_blank'>{row['ten_san_pham']}</a></h5>",
                unsafe_allow_html=True
            )

            # Hiển thị giá bán sản phẩm
            st.markdown(
                f"<p style='text-align: center; font-size: 16px; color: red; margin: 5px;'>"
                f"<b>Giá bán: {row['gia_ban']:,} đ</b></p>",
                unsafe_allow_html=True
            )

            # Hiển thị số sao đánh giá
            st.markdown(
                f"<p style='text-align: center; font-size: 14px; color: orange; margin: 5px;'>"
                f"⭐ {'⭐' * int(row['so_sao'])} ({row['so_sao']} sao)</p>",
                unsafe_allow_html=True
            )

            # Thêm khoảng cách giữa các dòng sản phẩm
            if i % 2 == 1:  # Sau mỗi 2 sản phẩm
                st.write("")  # Thêm một khoảng trắng
                st.markdown("---")  # Thêm một đường kẻ ngang để phân tách


# Tạo Tab
tab1_top, tab2_ground = st.tabs(["🛍️ Danh Sách Sản Phẩm có đánh giá cao nhất", "🛍️ Danh Sách Sản Phẩm có đánh giá thấp nhất"])

# Nội dung cho Tab 1: Sản phẩm có đánh giá cao nhất
with tab1_top:
    st.subheader("Sản phẩm được đánh giá cao nhất")
    num_images_top = st.session_state.get("num_images_high", 6)  # Số lượng hình ban đầu: 6
    
    highest_rated = combined_df.tail(10)  # Lấy 10 sản phẩm có đánh giá cao nhất
    display_images(highest_rated.head(num_images_top))  # Hiển thị sản phẩm
    
    # Kiểm tra nếu còn sản phẩm để load
    if num_images_top < len(highest_rated):
        if st.button("🔽 Xem thêm", key="high_more_tab1"):
            num_images_top += 6  # Tăng số lượng sản phẩm hiển thị thêm 6
            st.session_state.num_images_high = num_images_top
            st.rerun()
    else:
        st.write("🔔 Đã hiển thị tất cả sản phẩm!")

    # Nút "Thu gọn" (reset về 6 sản phẩm)
    if num_images_top > 6:
        if st.button("🔼 Thu gọn", key="low_collapse_tab1"):
            num_images_top = 6  # Reset về 6 sản phẩm
            st.session_state.num_images_high = num_images_top  # Cập nhật lại trong session_state
            st.rerun()  # Reload lại giao diện



# Nội dung cho Tab 2: Sản phẩm có đánh giá thấp nhất
with tab2_ground:
    st.subheader("Sản phẩm được đánh giá thấp nhất")
    num_images_ground = st.session_state.get("num_images_low", 6)  # Số lượng hình ban đầu: 6

    lowest_rated = combined_df.head(10)  # Lấy 10 sản phẩm có đánh giá thấp nhất
    display_images(lowest_rated.head(num_images_ground))  # Hiển thị sản phẩm

    if num_images_ground < len(lowest_rated):
        if st.button("🔽 Xem thêm", key="high_more_tab2"):
            num_images_ground += 6  # Tăng số lượng sản phẩm hiển thị thêm 6
            st.session_state.num_images_low = num_images_ground
            st.rerun()  # Sử dụng st.rerun thay cho st.experimental_rerun()
    else:
        st.write("🔔 Đã hiển thị tất cả sản phẩm!")

    # Nút "Thu gọn" (reset về 6 sản phẩm)
    if num_images_ground > 6:
        if st.button("🔼 Thu gọn", key="low_collapse_tab2"):
            num_images_ground = 6  # Reset về 6 sản phẩm
            st.session_state.num_images_low = num_images_ground  # Cập nhật lại trong session_state
            st.rerun()  # Reload lại giao diện


########################## WORDCLOUD NỘI DUNG BÌNH LUẬN TRỨC VÀ SAU KHI XỬ LÝ
# HÀM VẼ WORDCLOUD
def generate_wordcloud_and_top_words(text, stopwords=None, slider_key="slider"):
    """
    Tạo Word Cloud và trả về từ điển chứa các từ phổ biến nhất.
    """
    # Tính tần suất từ
    words = text.split()
    word_counts = Counter(words)

    # Loại bỏ stopwords (nếu có)
    if stopwords:
        word_counts = Counter({word: count for word, count in word_counts.items() if word not in stopwords})

    # Chỉ giữ lại các từ phổ biến nhất thông qua slider (thêm key để tránh lỗi)
    num_words = st.slider("Chọn số lượng từ phổ biến để hiển thị", min_value=5, max_value=50, value=10, step=1, key=slider_key)
    top_words = word_counts.most_common(num_words)
    top_words_dict = dict(top_words)

    # Trả về Word Cloud và danh sách top từ phổ biến
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words_dict)
    return wordcloud, top_words
#----------

# Tạo hai tab cho WORD CLOUD
st.header('6. WordCloud CHO BÌNH LUẬN')

# Nối tất cả các bình luận thành một văn bản
text = " ".join(df['noi_dung_binh_luan'].dropna())

# Gọi hàm và chia tab
wordcloud, top_words = generate_wordcloud_and_top_words(text, slider_key="slider")
tab1, tab2 = st.tabs(["Word Cloud ", "Top Từ Phổ Biến"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

with tab2:
    for word, count in top_words:
        st.write(f"{word}: {count}")


##################### WC POSITIVE, NEGATIVE WORDS

# POSITIVE
st.subheader("Word Cloud và Top Từ POSITIVE Phổ Biến")

# Load POSITIVE_WORDS
positive_words = load_and_process_file('files/positive_words_VN.txt')
print("Count of positive_words:", len(positive_words))

# Chuyển đổi key, thay khoảng trắng bằng '_'
list_positive_words = [key.replace(' ', '_') for key in positive_words.keys()]

# Chuyển danh sách thành một chuỗi văn bản
text_positive_words = ' '.join(list_positive_words)

# Load NEGATIVE_WORDS
negative_words = load_and_process_file('files/negative_words_VN.txt')
print("Count of negative_words:", len(negative_words))

# Chuyển đổi key, thay khoảng trắng bằng '_'
list_negative_words = [key.replace(' ', '_') for key in negative_words.keys()]

# Chuyển danh sách thành một chuỗi văn bản
text_negative_words = ' '.join(list_negative_words)

# Gọi hàm và chia tab
wordcloud_positive, top_words_positive = generate_wordcloud_and_top_words(text_positive_words, slider_key="slider_positive")
tab1_positive, tab2_positive = st.tabs(["Word Cloud POSITIVE", "Top Từ Phổ Biến POSITIVE"])

with tab1_positive:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_positive, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

with tab2_positive:
    for word, count in top_words_positive:
        st.write(f"{word}: {count}")

# NEGATIVE
st.subheader("Word Cloud và Top Từ NEGATIVE Phổ Biến")

# Gọi hàm và chia tab
wordcloud_negative, top_words_negative = generate_wordcloud_and_top_words(text_negative_words, slider_key="slider_negative")
tab1_negative, tab2_negative = st.tabs(["Word Cloud NEGATIVE", "Top Từ Phổ Biến NEGATIVE"])

with tab1_negative:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_negative, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

with tab2_negative:
    for word, count in top_words_negative:
        st.write(f"{word}: {count}")


###################
# Inside the Data: Bên trong dữ liệu.
# Data Unveiled: Hé lộ dữ liệu.
# Beneath the Numbers: Dưới những con số.
# Deep Dive into Data: Đi sâu vào dữ liệu.
# Cracking the Data Code: Giải mã dữ liệu.
