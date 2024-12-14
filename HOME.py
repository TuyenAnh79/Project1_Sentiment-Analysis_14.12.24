import streamlit as st

st.set_page_config(
    page_title="Trang chính",
    page_icon="favicon.png",
)
# Đặt hình ảnh header
st.image("banner.jpg")#,  use_container_width=True

st.write("# Welcome to PROJECT 1 👋")

st.sidebar.success("Giáo Viên Hướng Dẫn: \n # KHUẤT THUỲ PHƯƠNG")
st.sidebar.success("Học Viên:\n # NGUYỄN CHẤN NAM \n # CHẾ THỊ ANH TUYỀN")
st.sidebar.success("Ngày báo cáo: \n # 16/12/2024")


st.markdown(
    """
    ### *SUBJECT:*
    # HASAKI.VN - SENTIMENT ANALYSIS 
    ### Business Objective/Problem
    #### * “HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn...
        
    #### * Khách hàng có thể lên đây để lựa chọn sản phẩm,xem các đánh giá/ nhận xét cũng như đặt mua sản phẩm.

    #### Mục tiêu của bài toán: Xây dựng model dể dự đoán khách hàng có hài lòng với sản phẩm hay không dựa vào comment khách hàng để lại trên trang web

    ### Danh mục các việc cần thực hiện:
"""
)

# Tạo hyperlink thủ công
st.markdown("##### [- Data Exploration 🔍](https://project1sentiment-analysis141224-tyhzdnzwawgx29crzsfflt.streamlit.app/Data_Exploration)", unsafe_allow_html=True)
st.markdown("##### [- Models 📊](https://project1sentiment-analysis141224-tyhzdnzwawgx29crzsfflt.streamlit.app/Models)", unsafe_allow_html=True)
st.markdown("##### [- New Predict 🎯](https://project1sentiment-analysis141224-tyhzdnzwawgx29crzsfflt.streamlit.app/New_Predict)", unsafe_allow_html=True)
st.markdown("##### [- Login 🔑](https://project1sentiment-analysis141224-tyhzdnzwawgx29crzsfflt.streamlit.app/Login)", unsafe_allow_html=True)
st.markdown("##### [- For_Production_House 🔍](https://project1sentiment-analysis141224-tyhzdnzwawgx29crzsfflt.streamlit.app/For_Production_House)", unsafe_allow_html=True)





