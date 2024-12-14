import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar
st.sidebar.success("Giáo Viên Hướng Dẫn: \n # KHUẤT THUỲ PHƯƠNG")
st.sidebar.success("Học Viên:\n # NGUYỄN CHẤN NAM \n # CHẾ THỊ ANH TUYỀN")
st.sidebar.success("Ngày báo cáo: \n # 16/12/2024")


# Đọc dữ liệu sản phẩm và đánh giá
san_pham = pd.read_csv('data/San_pham.csv', index_col='ma_san_pham')
danh_gia = pd.read_csv('data/Danh_gia.csv', index_col=0)
khach_hang = pd.read_csv("data/Khach_hang.csv")

# Hàm phân loại dựa trên giá trị của cột 'so_sao'
def classify_rating(star_rating):
    if star_rating <= 4:
        return 'negative'
    elif star_rating == 5:
        return 'positive'

# Áp dụng hàm vào cột 'so_sao' để tạo cột mới 'phan_loai_danh_gia'
danh_gia['phan_loai_danh_gia'] = danh_gia['so_sao'].apply(classify_rating)

# Kết hợp dữ liệu đánh giá và sản phẩm
danhgia_sanpham = danh_gia.merge(san_pham, on="ma_san_pham", how='left')
danhgia_sanpham = danhgia_sanpham.merge(khach_hang, on='ma_khach_hang', how='left')
df_reviews = danhgia_sanpham[['ma_khach_hang','ho_ten', 'ma_san_pham', 'ten_san_pham', 'gia_ban',
                              'ngay_binh_luan', 'gio_binh_luan', 'noi_dung_binh_luan',
                              'phan_loai_danh_gia', 'so_sao']]

# Đọc dữ liệu từ file Khach_hang.csv
def load_users(df):
    df.columns = df.columns.str.strip()
    df["ma_khach_hang"] = df["ma_khach_hang"].astype(str).str.strip()
    df["mat_khau"] = df["mat_khau"].astype(str).str.strip()
    return df

# Hiển thị thông tin đánh giá của khách hàng
def display_customer_reviews(maKH, df_reviews):
    filtered_reviews = df_reviews[df_reviews['ma_khach_hang'] == maKH]
    filtered_reviews = filtered_reviews[['ngay_binh_luan', 'gio_binh_luan','ma_san_pham', 'ten_san_pham', 'gia_ban',
                                         'noi_dung_binh_luan','phan_loai_danh_gia', 'so_sao']]
    filtered_reviews['ngay_binh_luan'] = pd.to_datetime(filtered_reviews['ngay_binh_luan'], errors='coerce')  # Đảm bảo 'ngay_binh_luan' có kiểu dữ liệu ngày tháng
    filtered_reviews = filtered_reviews.sort_values(by='ngay_binh_luan', ascending=False)  # Sắp xếp giảm dần theo ngày
    
    # Hiển thị số lượng đánh giá
    num_reviews = len(filtered_reviews)
    num_products = filtered_reviews['ma_san_pham'].nunique()
    
    # # Hiển thị số sao
    # star_count = filtered_reviews['so_sao'].value_counts().sort_index().reset_index()
    # star_count.columns = ['so_sao', 'count']  # Đổi tên cột thành 'so_sao' và 'count'
    # # Thay đổi giá trị trong cột 'so_sao' thành 'x sao'
    # star_count['so_sao'] = star_count['so_sao'].astype(str) + ' sao'
    # # Đặt 'so_sao' làm chỉ mục và chuyển thành dạng transpose
    # star_count = star_count.set_index('so_sao').T

    
    
    # Hiển thị phân loại đánh giá (positive/negative)
    positive_count = filtered_reviews[filtered_reviews['phan_loai_danh_gia'] == 'positive'].shape[0]
    negative_count = filtered_reviews[filtered_reviews['phan_loai_danh_gia'] == 'negative'].shape[0]
    
    # Hiển thị thông tin tổng quát
    st.write(f"Bạn đã đánh giá: {num_reviews} lần.")
    st.write(f"Tổng sản phẩm bạn đã đánh giá: {num_products} sản phẩm.")
#---------- số sao đánh giá
    # Thay đổi giá trị sao thành chuỗi và đếm số lượng sao
    star_count = filtered_reviews['so_sao'].value_counts().sort_index().reset_index()
    star_count.columns = ['so_sao', 'count']

    # Chuyển đổi các giá trị sao thành chuỗi và thêm " sao"
    star_count['so_sao'] = star_count['so_sao'].astype(str) + ' sao'

    # Đảm bảo rằng có đầy đủ 5 mức sao từ 1 sao đến 5 sao, nếu thiếu, thêm các mức sao có giá trị count = 0
    all_stars = ['1 sao', '2 sao', '3 sao', '4 sao', '5 sao']
    star_count = star_count.set_index('so_sao').reindex(all_stars, fill_value=0).reset_index()

    # Vẽ biểu đồ barchart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(star_count['so_sao'], star_count['count'], color='skyblue')
    ax.set_xlabel('Số sao')
    ax.set_ylabel('Số lượng')
    ax.set_title('Số lượng đánh giá theo mức sao')

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
#---------------
    
    st.write(f"Số đánh giá tích cực: {positive_count}, số đánh giá tiêu cực: {negative_count}")
    
    # Biểu đồ phân phối giá các sản phẩm bạn đã đánh giá
    plt.figure(figsize=(8, 6))
    filtered_reviews.groupby('ma_san_pham')['gia_ban'].mean().plot(kind='bar', color='lightblue')
    plt.title("Phân phối giá các sản phẩm đã đánh giá")
    plt.xlabel("Sản phẩm")
    plt.ylabel("Giá trung bình")
    plt.xticks(rotation=45)
    # Hiển thị biểu đồ
    st.pyplot(plt)

        # Biểu đồ phân phối đánh giá theo ngày
    plt.figure(figsize=(10, 6))
    # Chuyển cột 'ngay_binh_luan' về dạng datetime nếu chưa có
    filtered_reviews['ngay_binh_luan'] = pd.to_datetime(filtered_reviews['ngay_binh_luan'], errors='coerce')
    # Nhóm theo ngày và đếm số lượng đánh giá mỗi ngày
    date_counts = filtered_reviews.groupby(filtered_reviews['ngay_binh_luan'].dt.date).size()
    # Vẽ biểu đồ
    date_counts.plot(kind='line', marker='o', color='orange')
    plt.title("Phân phối đánh giá theo ngày")
    plt.xlabel("Ngày")
    plt.ylabel("Số lượng đánh giá")
    plt.xticks(rotation=45)  # Nghiêng nhãn ngày 45 độ
    # Hiển thị biểu đồ
    st.pyplot(plt)

#----------------------

    # Tiêu đề ứng dụng
    st.title("Tìm kiếm thông tin đánh giá sản phẩm")

    # Tuỳ chọn phương thức tìm kiếm
    search_option = st.radio("Chọn phương thức tìm kiếm:", ["Chọn từ danh sách", "Tìm theo từ khóa"])

    # Cách 1: Tìm kiếm bằng Selectbox
    if search_option == "Chọn từ danh sách":
        search_product = st.selectbox("Thông tin sản phẩm đã đánh giá", filtered_reviews['ten_san_pham'].unique())
        product_reviews = filtered_reviews[filtered_reviews['ten_san_pham'] == search_product]

    # Cách 2: Tìm kiếm bằng từ khóa
    elif search_option == "Tìm theo từ khóa":
        keyword = st.text_input("Nhập từ khóa tìm kiếm sản phẩm")
        if keyword:  # Chỉ tìm kiếm nếu có từ khóa nhập vào
            product_reviews = filtered_reviews[filtered_reviews['ten_san_pham'].str.contains(keyword, case=False, na=False)]
        else:
            product_reviews = pd.DataFrame()  # Không có từ khóa thì kết quả rỗng

    # Kiểm tra nếu có sản phẩm đánh giá
    if not product_reviews.empty:
        st.write(f"Thông tin đánh giá cho sản phẩm:")

        # Chọn các cột cần thiết từ product_reviews
        product_reviews = product_reviews[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'ngay_binh_luan', 'noi_dung_binh_luan', 'phan_loai_danh_gia', 'so_sao']]

        # Data chứa link hình sản phẩm
        image_df = pd.read_csv('files_nam/San_pham_Link_Image_Brand.csv')

        # Kết hợp dữ liệu hình ảnh và thông tin sản phẩm
        product_reviews = pd.merge(product_reviews, image_df, on="ma_san_pham", how="inner")

        # # Hiển thị dữ liệu đã được kết hợp
        # st.dataframe(product_reviews)

        for index, row in product_reviews.iterrows():
            # Lấy thông tin sản phẩm
            product_name = row['ten_san_pham']
            product_link = row['chi_tiet']
            product_price = row['gia_ban']
            stars = row['so_sao']
            review_date = row['ngay_binh_luan']
            review_content = row['noi_dung_binh_luan']
            product_image_path = row['hinh_anh']  # Đảm bảo có đường dẫn tới hình ảnh sản phẩm

            # Tạo 2 cột trong Streamlit
            col1, col2 = st.columns([1, 3])  # Cột 1 (hình ảnh) và cột 2 (thông tin sản phẩm và bình luận)

            # Cột 1: Hiển thị hình ảnh sản phẩm
            # Cột 1: Hiển thị hình ảnh sản phẩm
            with col1:
                try:
                    # Hiển thị hình ảnh sản phẩm từ URL hoặc đường dẫn cục bộ
                    st.image(row['hinh_anh'], width=150)  # Hiển thị hình ảnh với chiều rộng 300px
                
                # Hiển thị số sao
                    st.write(f"**Số sao:** {'⭐' * stars}")
                                
                                
                except Exception as e:
                    st.write(f"Lỗi khi tải hình ảnh sản phẩm: {e}")

            # Cột 2: Hiển thị các thông tin sản phẩm và đánh giá
            with col2:
                # Hiển thị tên sản phẩm và link
                st.write(f"**Sản phẩm:** [{product_name}]({product_link})")  # Link sản phẩm
                st.write(f"**Giá bán:** {product_price,} VNĐ")

                # Hiển thị ngày và nội dung bình luận
                st.write(f"**Ngày bình luận:** {review_date}")
                st.write(f"**Nội dung bình luận:** {review_content}")

    else:
        st.write("Không tìm thấy sản phẩm nào phù hợp.")
#----------------------

# Giao diện Streamlit
def main():
    st.subheader("Hệ thống đăng nhập và tra cứu khách hàng")

    # Tải thông tin người dùng
    df_users = load_users(khach_hang)
    VALID_USERS = df_users.set_index("ma_khach_hang")["mat_khau"].to_dict()

    # Khởi tạo trạng thái phiên
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["ho_ten"] = ""

    # Nếu chưa đăng nhập
    if not st.session_state["logged_in"]:
        st.header("Đăng nhập")
        username = st.text_input("👤 Tên đăng nhập (Mã khách hàng):").strip()
        password = st.text_input("🔑 Mật khẩu (Gợi ý: password123):", type="password").strip()
        login_button = st.button("Đăng nhập")

        if login_button:
            # Kiểm tra thông tin đăng nhập
            if username in VALID_USERS and VALID_USERS[username] == password:
                # Lấy thông tin tên khách hàng (ho_ten)
                ho_ten = df_users[df_users["ma_khach_hang"] == username]["ho_ten"].values[0]
                # Cập nhật session state với thông tin khách hàng
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["ho_ten"] = ho_ten
                st.success("Đăng nhập thành công!")
                st.rerun()  # Làm mới trang để vào trạng thái đăng nhập
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!")
    else:
        # Nếu đã đăng nhập
        st.markdown(f"##### Xin chào khách hàng: **{st.session_state['ho_ten']}** \n ##### Mã số khách hàng **{st.session_state['username']}**")

        # Hiển thị lịch sử đánh giá sản phẩm của khách hàng
        st.markdown("#### Lịch sử đánh giá sản phẩm")
        maKH = int(st.session_state["username"])  # Chuyển mã khách hàng về kiểu số nguyên
        display_customer_reviews(maKH, df_reviews)

        # Nút Logout
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["ho_ten"] = ""
            st.rerun()  # Làm mới giao diện để quay lại màn hình đăng nhập

if __name__ == "__main__":
    main()
