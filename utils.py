import re
import regrex
from underthesea import word_tokenize, pos_tag, sent_tokenize
import pandas as pd
from num2words import num2words
import re


################################ HÀM LOAD FILE TXT VÀ TẠO DICTIONARY

def load_from_file(file_path):
    """
    Load dữ liệu từ file và tạo dictionary.
    
    """
    result = {}

    with open(file_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()  # Loại bỏ khoảng trắng và ký tự xuống dòng
            if line:  # Chỉ xử lý dòng không trống
                if '\t' in line:  # Trường hợp dòng có tab
                    key, value = line.split('\t', 1)  # Tách key và value
                    result[key.lower()] = value.lower()  # Chuyển thành chữ thường
                else:  # Trường hợp không có tab, dùng dòng làm key và giá trị là rỗng
                    result[line.lower()] = ""  # Chuyển thành chữ thường

    return result

################################ HÀM LOAD FILE TXT VÀ TẠO VALUE TỪ KEY DICTIONARY

# Hàm đọc các file và tạo dictionary cho các từ điển
def load_and_process_file(file_path):
    
    result_dict = {}

    with open(file_path, 'r', encoding="utf8") as file:
        for line in file:
            # Bỏ qua dòng trống
            if not line.strip():
                continue

            # Lấy từ và thay khoảng trắng bằng dấu gạch dưới
            key = line.strip().lower()  # Key là từ gốc, chuyển thành chữ thường
            value = key.replace(" ", "_")  # Tạo value bằng cách thay thế khoảng trắng bằng "_"

            # Thêm vào dictionary
            result_dict[key] = value

    return result_dict

################################ HÀM THAY THẾ CÁC CỤM TỪ

def replace_phrases(text, dicts):
    """
    Thay thế các cụm từ teencode, emoji, dòng sản phẩm, brand thành các cụm từ thay thế
    """
    # Tokenize văn bản, chuyển thành chữ thường
    tokens = word_tokenize(text.lower(), format="text").split()
    new_tokens = []

    # Loại bỏ các ký tự không cần thiết (như các khoảng trắng dư thừa)
    tokens = [token.strip() for token in tokens if token.strip() != '']

    # Duyệt qua các từ điển để thay thế các cụm từ
    for current_dict in dicts:
        sorted_phrases = sorted(current_dict.items(), key=lambda x: len(x[0].split()), reverse=True)

        # Duyệt qua từng từ trong tokens
        i = 0
        while i < len(tokens):
            merged = False

            for phrase, replacement in sorted_phrases:
                phrase_tokens = phrase.split()
                phrase_length = len(phrase_tokens)

                # Kiểm tra nếu các token liên tiếp khớp với cụm từ
                if tokens[i:i + phrase_length] == phrase_tokens:
                    new_tokens.append(replacement)
                    i += phrase_length
                    merged = True
                    break

            if not merged:
                new_tokens.append(tokens[i])
                i += 1

    # Kết quả sau khi thay thế
    result = " ".join(new_tokens)
    result = result.replace("_", " ").strip()
    result = " ".join(result.split())

    return result

################################ HÀM THAY THẾ SỐ BẰNG CHỮ

def replace_units_with_words(text):
    """
    Thay thế các số đi kèm 'k', 'ml', 'g' và số không kèm đơn vị thành chữ,
    đồng thời nối các từ đã chuyển thành chữ lại với nhau bằng dấu '_'.
    """
    # Biểu thức chính quy và hàm thay thế
    patterns = [
        # Xử lý 'k' (k có hoặc không có khoảng trắng trước nó)
        (r'(\d+)\s?k', lambda match: "_".join(num2words(int(match.group(1)) * 1000, lang='vi').split()) + '_ngàn'),
        (r'(\d+)(?=\s?k)', lambda match: "_".join(num2words(int(match.group(1)) * 1000, lang='vi').split()) + '_ngàn'),
        
        # Xử lý 'ml' (liền kề hoặc có khoảng trắng)
        (r'(\d+)\s?ml', lambda match: "_".join(num2words(int(match.group(1)), lang='vi').split()) + '_mili_lít'),
        (r'(\d+)(?=\s?ml)', lambda match: "_".join(num2words(int(match.group(1)), lang='vi').split()) + '_mili_lít'),
        
        # Xử lý 'g' (liền kề hoặc có khoảng trắng)
        (r'(\d+)\s?g', lambda match: "_".join(num2words(int(match.group(1)), lang='vi').split()) + '_gram'),
        (r'(\d+)(?=\s?g)', lambda match: "_".join(num2words(int(match.group(1)), lang='vi').split()) + '_gram'),
        
        # Xử lý số không đi kèm đơn vị
        (r'(\d+)', lambda match: "_".join(num2words(int(match.group(1)), lang='vi').split()))
    ]
    
    # Thay thế các khớp tìm thấy trong văn bản theo từng pattern
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    return text

################################ HÀM CHUẨN HOÁ UNOCODE TIẾNG VIỆT

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

################################# HÀM XỬ LÝ CÁC TỪ TRONG DANH SẮCH ĐẶC BIỆT

# Hàm xử lý các từ trong danh sách đặc biệt (từ phủ định)
def process_special_word(text, special_words):
    # Chuyển danh sách đặc biệt thành chữ thường và thành set để tối ưu tìm kiếm
    special_words_set = set([word.lower() for word in special_words])  
    new_text = ''
    text_lst = text.lower().split()
    i = 0

    while i < len(text_lst):
        word = text_lst[i]
        if word in special_words_set:  # Kiểm tra nếu từ nằm trong danh sách đặc biệt
            next_idx = i + 1
            if next_idx < len(text_lst):  # Kiểm tra từ tiếp theo có tồn tại
                word = word + '_' + text_lst[next_idx]  # Ghép từ đặc biệt với từ tiếp theo
                i = next_idx + 1  # Bỏ qua từ tiếp theo
            else:
                i += 1  # Di chuyển tiếp nếu không có từ sau
        else:
            i += 1  # Tiếp tục với từ hiện tại
        new_text += word + ' '  # Ghép từ vào chuỗi mới

    return new_text.strip()

special_words = ["bị", "không", 'thoải','dễ','thư', 'sử','tột' ,'da','nhạy','chống','sản','kích','tẩy', 'mùi',
                'thứ','sản','gây','tẩy','hỗn','chân', 'cẩn','đầu','mụn','bị','khá','hơi','lên',
                 'đáng để','mặt','dễ', 'giao','giá','rất','dầu','sữa','nhạy','trang','hợp','hiệu','thiên','hỗn',
                 'bào','tế','tiếp','hai','mức','cảm', 'thấm', 'chữ', 'chính', 'bông', 'ủng', 'hàng', 'cải',
                'đều', 'cấp', 'giảm', 'thấy', 'bị', 'chất', 'cực', 'sữa', 'lại', 'nâng','nổi','đang', 'sữa', 'gel'] #,'rửa'

# Loại bỏ phần tử trùng lặp và sắp xếp theo bảng chữ cái
special_words = sorted(set(special_words))

#################### CHUẨN HOÁ VÀ GIỮ NGUYÊN CỤM TỪ GHÉP VÀ LOẠI BỎ KÝ TỰ ĐẶC BIỆT

def process_text(text):
    """
    Xử lý văn bản: chuẩn hóa, giữ nguyên cụm từ ghép và loại bỏ ký tự đặc biệt
    
    Args:
        text (str): Văn bản đầu vào
    
    Returns:
        str: Văn bản đã được xử lý và chuẩn hóa
    """
    # Chuẩn hóa văn bản: chữ thường và dấu câu
    document = text.lower()
    document = document.replace("\u2019", '')  # Loại bỏ dấu nháy đặc biệt
    document = re.sub(r'\.+', ".", document)  # Chuẩn hóa dấu chấm lặp
    
    new_sentence = ''

    # Xử lý từng câu trong văn bản
    for sentence in sent_tokenize(document):
        # 1. Tokenize bằng Underthesea để giữ cụm từ ghép
        tokens = word_tokenize(sentence, format="text").split()

        # 2. Loại bỏ ký tự đặc biệt, nhưng giữ khoảng trắng giữa các từ
        cleaned_tokens = [re.sub(r'[^a-z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', '', token)
                          for token in tokens]

        # 3. Gộp lại thành câu đã xử lý
        sentence = ' '.join(token for token in cleaned_tokens if token)
        new_sentence += sentence + '. '

    # Chuẩn hóa khoảng trắng và dấu câu
    document = re.sub(r'\s+', ' ', new_sentence).strip()
    return document

# Kiểm thử
text = 'sau bảy mươi bảy bốn mươi chín dòng sữa rửa mặt , thì chân ái là đây , mua tuyp nhỏ dùng thử trước , thấy tốt nên nay mua chai bốn trăm mili lít luôn , nay là chai thứ hai rồi mới lên đây đánh giá . nói chung ai da nhờn , khô , hay hở tí là nổi mụn thì chai này ok nha , một trăm điểm'
output = process_text(text)
print(output)

################################# HÀM CHUẨN HOÁ CÁC TỪ CÓ KÝ TỰ LẶP

# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

################################# HÀM XỬ LÝ CỤM TỪ, LOẠI TỪ TRONG VĂN BẢN

def process_postag_thesea(text, special_words):
    if not isinstance(text, str) or not text.strip():
        return ''
    
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.', '')  # Loại bỏ dấu chấm
        
        lst_word_type = ['N', 'Np', 'A', 'V']  # Các loại từ cần giữ lại #, 'AB', 'VB', 'VY', 'R'
        
        processed_words = []
        for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"), special_words)):
            word_text = word[0].strip()
            word_type = word[1].upper().strip()
            
            if word_type in lst_word_type:
                processed_words.append(word_text)
        
        sentence = ' '.join(processed_words)
        new_document += sentence + ' '

    new_document = re.sub(r'\s+', ' ', new_document).strip()
    return new_document

################################# HÀM REMOVE STOPWORD

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = re.sub(r'\s+', ' ', document).strip()
    return document

#################################
from wordcloud import WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st

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

################################# HÀM TIỀN XỬ LÝ VĂN BẢN

# Hàm tiền xử lý văn bản
def preprocessing(text, brand_dict,teen_dict, emoji_dict,dong_san_pham_dict,
                  positive_words,negative_words,tu_ghep_dict, special_words, stopwords_lst):
    # Bắt đầu với text ban đầu
    text_pre = text

    # Replace text bằng nội dung trong các thư viện
    text_pre = replace_phrases(text_pre, [brand_dict])
    text_pre = replace_phrases(text_pre, [teen_dict])
    text_pre = replace_phrases(text_pre, [emoji_dict])
    text_pre = replace_phrases(text_pre, [dong_san_pham_dict])
    text_pre = replace_phrases(text_pre, [positive_words])
    text_pre = replace_phrases(text_pre, [negative_words])
    text_pre = replace_phrases(text_pre, [tu_ghep_dict])

    # Thay đơn vị đo lường bằng chữ
    text_pre = replace_units_with_words(text_pre)
    
    # Chuẩn hoá unicode
    text_pre = covert_unicode(text_pre)
    
    # Xử lý các từ đặc biệt
    text_pre = process_special_word(text_pre, special_words)
    
    # Chuẩn hóa, giữ nguyên cụm từ ghép và loại bỏ ký tự đặc biệt
    text_pre = process_text(text_pre)
    
    # Hàm để chuẩn hóa các từ có ký tự lặp
    text_pre = normalize_repeated_characters(text_pre)
    
    # Hàm xử lý cụm từ, loại từ trong văn bản
    text_pre = process_postag_thesea(text_pre, special_words)
    
    # Remove stopwords
    text_pre = remove_stopword(text_pre, stopwords_lst)
    
    return text_pre
