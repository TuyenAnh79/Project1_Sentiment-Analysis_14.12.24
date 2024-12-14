import pickle
import streamlit as st
import pandas as pd
import pickle
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


st.set_page_config(page_title="New Predict", page_icon="🎯")

# Sidebar
st.sidebar.success("Giáo Viên Hướng Dẫn: \n # KHUẤT THUỲ PHƯƠNG")
st.sidebar.success("Học Viên:\n # NGUYỄN CHẤN NAM \n # CHẾ THỊ ANH TUYỀN")
st.sidebar.success("Ngày báo cáo: \n # 16/12/2024")

################################ LOAD CÁC FILE HỖ TRỢ VIỆC PREPROCESSING DATA

# Load TEENCODE và sắp xếp theo chiều dài key giảm dần
teen_dict = load_from_file('files/teencode.txt')
teen_dict = dict(sorted(teen_dict.items(), key=lambda item: len(item[0]), reverse=True))

# Load EMOJICON
emoji_dict = load_from_file('files/emojicon.txt')

# Load STOPWORDS
stopwords_dict = load_from_file('files/vietnamese-stopwords.txt')
stopwords_lst = list(stopwords_dict.keys())

# Load WRONG WORDS và tự động loại bỏ trùng lặp
wrong_dict = load_from_file('files/wrong-word.txt')
wrong_lst=list(wrong_dict.keys())

# Load measurement_unit
measurement_unit = load_from_file('files/measurement_unit.txt')
#-------

# Load tu_ghep_dict
tu_ghep_dict = load_and_process_file('files/tu_ghep.txt')

# Load dong_san_pham_dict
dong_san_pham_dict = load_and_process_file('files/dong_san_pham.txt')

# Load brand_dict
brand_dict = load_and_process_file('files/brand.txt')

# Load POSITIVE_WORDS
positive_words = load_and_process_file('files/positive_words_VN.txt')

# Load NEGATIVE_WORDS
negative_words = load_and_process_file('files/negative_words_VN.txt')


################################ DỰ ĐOÁN GIÁ TRỊ MỚI


# Hàm dự đoán giá trị mới cho các bình luận
def predict_new_data_with_probabilities(model, new_texts, vectorizer):
    
    # Chuyển đổi văn bản mới thành vector
    new_texts_transformed = vectorizer.transform(new_texts)
    
    # Dự đoán xác suất cho các lớp (positive và negative)
    probabilities = model.predict_proba(new_texts_transformed)
    
    # Dự đoán nhãn cuối cùng
    predictions = model.predict(new_texts_transformed)
    
    # Kết quả: Xác suất và nhãn dự đoán
    results = []
    for text, prob, pred in zip(new_texts, probabilities, predictions):
        sentiment = "positive" if pred == 1 else "negative"
        results.append({
            "Bình luận": text,
            "Xác suất Positive": round(prob[1], 4),  # Xác suất lớp 1
            "Xác suất Negative": round(prob[0], 4),  # Xác suất lớp 0
            "Kết quả": sentiment  # Kết quả cuối cùng
        })
    return results


# Tải mô hình và vectorizer từ các file pickle
model_filename = 'logistic_regression_model.pkl'  
vectorizer_filename = 'logistic_regression_vectorizer.pkl'


# Ứng dụng Streamlit
st.title("🔮 Sentiment Prediction")
st.write("Dự đoán cảm xúc (tích cực/tiêu cực) của bình luận dựa trên mô hình học máy.")

try:
    # Tải mô hình LOGISTIC REGRESSION và vectorizer từ file
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_filename, 'rb') as f:
        vectorizer = pickle.load(f)

    # Giao diện nhập dữ liệu
    st.subheader("📝 Nhập bình luận để dự đoán:")

    # Chọn phương thức nhập liệu: nhập tay hoặc tải file
    option = st.radio("Chọn phương pháp nhập liệu:", ["Nhập văn bản trực tiếp", "Tải file văn bản (.txt hoặc .csv)"])

    # Biến để lưu danh sách bình luận
    comments = []

    if option == "Nhập văn bản trực tiếp":
        # Nhập văn bản
        new_text = st.text_area("Nhập bình luận, mỗi bình luận trên một dòng:")

        st.markdown('**Nội dung bình luận mẫu:**')
        st.markdown('"Đã mua đủ màu, rồi đổi sp khác nhưng vẫn phải quay lại với màu Hồng, dùng màu Hồng đi mn ơi, rất mềm mịn da, dùng 2-3 ngày thấy ngay khác biệt"')

        if new_text.strip():
            comments = new_text.splitlines()

    elif option == "Tải file văn bản (.txt hoặc .csv)":
        uploaded_file = st.file_uploader("Tải file bình luận (.txt hoặc .csv):", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.txt'):
                # Đọc file .txt
                content = uploaded_file.read().decode('utf-8')
                comments = content.splitlines()
            elif uploaded_file.name.endswith('.csv'):
                # Đọc file .csv
                df = pd.read_csv(uploaded_file)
                if 'comment' in df.columns:  # Kiểm tra cột dữ liệu
                    comments = df['comment'].dropna().tolist()
                else:
                    st.error("File CSV cần có cột tên 'comment' chứa nội dung bình luận!")

    if st.button("🎯 Dự đoán"):
        if comments:
            try:
                # Tiền xử lý các bình luận
                preprocessed_comments = [
                    preprocessing(comment, brand_dict, teen_dict, emoji_dict, dong_san_pham_dict,
                                  positive_words, negative_words, tu_ghep_dict, special_words, stopwords_lst)
                    for comment in comments
                ]
            
                # Biến đổi dữ liệu mới bằng vectorizer đã fit
                X_new = vectorizer.transform(preprocessed_comments)

                # Dự đoán kết quả và xác suất
                predictions = model.predict(X_new)
                probabilities = model.predict_proba(X_new)

                # Tạo DataFrame kết quả
                results_df = pd.DataFrame({
                    'Bình luận': comments,
                    'Dự đoán': predictions,
                    'Xác suất Lớp 0': probabilities[:, 0],
                    'Xác suất Lớp 1': probabilities[:, 1]
                })

                # Hiển thị kết quả dưới dạng bảng
                st.subheader("🔍 Kết quả Dự đoán:")
                st.table(results_df)

                # Cho phép tải file kết quả về
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💽 Tải kết quả dự đoán về file CSV",
                    data=csv,
                    file_name="sentiment_predictions_with_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi xử lý bình luận: {e}")
        else:
            st.warning("Vui lòng nhập bình luận hoặc tải file để dự đoán!")

except FileNotFoundError:
    st.error(f"Không tìm thấy file '{model_filename}' hoặc '{vectorizer_filename}'. Vui lòng kiểm tra lại đường dẫn file.")
except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
