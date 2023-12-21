import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pyvi import ViTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk

# Thêm sidebar
st.sidebar.title('Filter')

# Thêm một widget trong sidebar để nhập dữ liệu
user_input = st.sidebar.text_input('Nhập Id User (nếu có):', '')
user_password = st.sidebar.text_input('Nhập password:', type='password')

# Thêm một widget trong sidebar để chọn một màu
selected_color = st.sidebar.color_picker('Chọn màu background:', '#c1eac1')

# Kéo thả hai thanh slider để chọn khoảng cách
sidebar_max_price = 10000000
sidebar_min_price = st.sidebar.slider('Giá tối thiểu:',   min_value=0, max_value=sidebar_max_price - 1, value=0)
sidebar_max_price = st.sidebar.slider('Giá tối đa:', min_value=sidebar_min_price, max_value=10000000, value=10000000)

formatted_sidebar_min_price = '{:,}'.format(sidebar_min_price)
formatted_sidebar_max_price = '{:,}'.format(sidebar_max_price)
st.sidebar.write('Khoảng giá bạn đã chọn:')
st.sidebar.write(f'{formatted_sidebar_min_price} VND - {formatted_sidebar_max_price} VND')

list_location = ['Đà Lạt', 'Đà Nẵng', 'Hội An', 'Huế', 'Nha Trang', 'Phan Thiết', 'Phú Quốc', 'Quy Nhơn', 'Vũng Tàu']
sidebar_selected_location = st.sidebar.multiselect('Chọn địa điểm:', list_location, default=['Phú Quốc'])

sidebar_min_star, sidebar_max_star = st.sidebar.slider("Chọn khoảng sao phù hợp:", min_value=0, max_value=5, value=(0, 5))
sidebar_min_Rating, sidebar_max_Rating = st.sidebar.slider("Chọn khoảng điểm đánh giá:", min_value=0, max_value=10, value=(0, 10))
sidebar_min_CountRating, sidebar_max_CountRating = st.sidebar.slider("Chọn khoảng số lượng đánh giá:", min_value=0, max_value=2500, value=(0, 2500))
sidebar_min_distance, sidebar_max_distance = st.sidebar.slider("Chọn khoảng cách trung tâm (m):", min_value=0, max_value=27000, value=(0, 27000))


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {selected_color};
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


st.title('HỆ THỐNG RECOMMENDED HOTEL')
st.write('Hệ thống Recommended Hotel là hệ thống đề xuất khách sạn dựa trên dữ liệu khách hàng và thông tin thị trường. Hệ thống sử dụng các thuật toán CF và CB để đưa ra các đề xuất phù hợp với nhu cầu của từng khách hàng.')
st.write('Hệ thống hoạt động với 3 tác vụ chính:')
st.write('- Cho người mới bắt đầu.')
st.write('- Cho người đã có tài khoản dựa vào lịch sử.')
st.write('- Cho người đã có tài khoản dựa vào lịch sử và các tùy chọn mới (Hybrid-based).')

with st.expander('=== Bạn đã có tài khoản ==='):
    st.write('Hãy nhập thông tin tài khoản của bạn và lọc filter.')

with st.expander('== Bạn chưa có tài khoản =='):
    # Tạo range slider cho khoảng giá trị
    min_price, max_price = st.slider("Chọn khoảng giá phù hợp:", min_value=0, max_value=8000000, value=(0, 1000000))

    formatted_min_price = '{:,}'.format(min_price)
    formatted_max_price = '{:,}'.format(max_price)
    st.write(f'Khoảng giá bạn đã chọn: {formatted_min_price} VND - {formatted_max_price} VND')

    def get_price_level(min_price, max_price):
        levels = []
        
        if max_price < 400000:
            levels.append('Thấp')
        elif max_price <= 1000000:
            if min_price >= 400000: levels.append('Trung bình')
            else: levels.extend(['Thấp', 'Trung bình'])
        else:
            if min_price > 1000000: levels.append('Cao')
            elif min_price >= 400000: levels.extend(['Trung bình', 'Cao'])
            else: levels.extend(['Thấp', 'Trung bình', 'Cao'])
        
        return levels

    levels = get_price_level(min_price, max_price)
    facilities = ['Trung tâm thể dục', 'Máy pha trà/cà phê trong tất cả các phòng', 'Hồ bơi', 'Chỗ đỗ xe', 'Quầy bar', '2 hồ bơi', 'Tiện nghi cho khách khuyết tật', 'Lễ tân 24 giờ', 'Bữa sáng tuyệt vời', 'Sân thượng / hiên', 'Bữa sáng tốt', 'Nhà hàng', 'Bữa sáng', 'Chỗ đậu xe riêng', 'Hồ bơi ngoài trời', 'Nhà hàng', 'WiFi', 'Lễ tân 24h', 'Sân vườan', '3 nhà hàng', 'WiFi miễn phí', 'Dịch vụ phòng', '3 hồ bơi', 'Điều hòa nhiệt độ', 'Giáp biển', '5 nhà hàng', 'Hồ bơi trong nhà', 'Xe đưa đón sân bay', 'WiFi nhanh miễn phí ', 'Sân vườn', 'Khu vực bãi tắm riêng', 'Phòng gia đình ', 'Thang máy', 'Phòng không hút thuốc', 'Tiện nghi BBQ', 'Dọn phòng hàng ngày', 'Dịch vụ đưa đón sân bay (miễn phí)', 'Chỗ đậu xe', 'Thang máy', 'Trung tâm Spa & chăm sóc sức khoẻ', 'Chỗ đậu xe (trong khuôn viên)', 'Bữa sáng xuất sắc', 'Bữa sáng tuyệt hảo', 'Máy lạnh', 'Chỗ đỗ xe miễn phí', 'Bữa sáng rất tốt', 'Phòng gia đình', '2 nhà hàng']
    around = ['Sân vận động', 'Sân golf', 'Hang', 'Suối', 'Phố đi bộ', 'Bảo tàng văn hóa', 'Nhà thờ', 'Phòng trưng bày', 'Sông', 'Khu du lịch', 'Đền', 'Chùa', 'Tượng Đài', ' Sân bóng đá', 'Núi', 'Tháp', 'Nhà hàng', 'Phòng tranh', 'Sở thú', 'Bảo tàng', 'Địa điểm nổi tiếng', 'Di tích', 'Hồ', 'Miếu', 'Trường', 'Miếu ', 'Nhà Thờ', 'Công viên', 'Cafe/quán bar', 'Biển', 'Khu giải trí', 'Sân bóng đá', 'Chợ đêm', 'Siêu thị', 'Vườn hoa', 'Chợ', 'Đình', 'Khu vui chơi', 'Khu cắm trại', 'chùa', 'Cầu', 'Quảng trường', 'Công Viên', 'Khu mua sắm', 'bảo tàng', 'Nhà ga', 'Di tích lịch sử', 'Bãi đá', 'Sân bay', 'Hải đăng', 'Bệnh viện', 'Lâu đài', 'Di tích ', 'Đồi cát', 'Bến xe', 'Tượng đài', 'Lăng', 'Biển']
    around2 = ['Hang', 'Tượng Đài', 'Núi', 'Phòng tranh', 'Sân Bay', 'Địa điểm nổi tiếng', 'Di tích', 'Miếu', 'Trường', 'Khu du lịch ', 'Sân bóng đá', 'Siêu thị', 'Đình', 'Đèo', 'Nhà ga', 'Bệnh viện', 'Hải đăng', 'Bãi biển', 'Thủy Cung', 'Xe Buýt', 'Phòng trưng bày', 'Xe buýt', 'Đền', 'Chùa', 'Nhà hàng', 'Sân Golf', 'Công viên', 'Khu giải trí', 'Vườn quốc gia', 'Làng chài', 'Vườn hoa', 'Khu vui chơi', 'Vườn Quốc gia', 'Tàu lửa', 'bảo tàng', 'Thủy cung', 'Đường hầm', 'Sông','Nhà tù', 'Tháp', 'Thác nước', 'Bảo tàng', 'Biển,', 'Suối nước nóng', 'Cánh đồng', ' Suối nước nóng', 'Bến xe', 'Bến Du Thuyền', 'Sân bóng', 'Biển', 'Đền thờ', 'Sân bay', 'Vườn Quốc Gia', 'Lăng', 'Sân vận động', 'Sân golf', 'Cánh đông', 'Suối', 'Nhà thờ', 'Khu du lịch', 'Sở thú', 'Khu giả trí', 'Hồ', 'Thác', 'Cafe/quán bar', 'Đồi', 'Địa điểm chụp hình', 'Chợ', 'Bảo tràng', 'Khu cắm trại', 'Cầu', 'Quảng trường', 'Công Viên', 'Khu mua sắm', 'Bãi đá', 'Lâu đài', 'Đồi cát', 'Viện Hải dương học', 'Tượng đài']

    # Cho phép người dùng chọn một trong ba mức độ


    selected_facilities = st.multiselect('Chọn tiện tích:', facilities)
    if len(selected_facilities) == 0:
        st.write('Hãy chọn ít nhất một tiện ích')

    selected_around = st.multiselect('Chọn địa điểm xung quanh (< 1km):', around)
    if len(selected_around) == 0:
        st.write('Hãy chọn ít nhất một địa điểm xung quanh')

    selected_around2 = st.multiselect('Chọn địa điểm lân cận (>= 1km):', around2)
    if len(selected_around2) == 0:
        st.write('Hãy chọn ít nhất một địa điểm lân cận')


    #-------------------Content Base-------------------
    hotel_data = pd.read_csv('data_info_hotel_new.csv')

    user_pri = [', '.join(levels)]
    user_fac = [', '.join(selected_facilities)]
    user_ar = [', '.join(selected_around)]
    user_vici = [', '.join(selected_around2)]


    hotel_fac_corpus = hotel_data['Facilities'].tolist()
    hotel_ar_corpus = hotel_data['Around'].tolist()
    hotel_vici_corpus = hotel_data['Vicinity'].tolist()
    hotel_pri_corpus = hotel_data['Price_Category'].tolist()

    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.replace(',', '')
        tokens = ViTokenizer.tokenize(text)
        tokens = tokens.lower().split()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def extract_document_vectors(corpus):
        preprocessed_corpus = [preprocess_text(text) for text in corpus]
        model = Word2Vec(sentences=preprocessed_corpus, vector_size=300, window=10, min_count=1)

        document_vectors = []
        for text in preprocessed_corpus:
            doc_vector = []
            for word in text:
                if word in model.wv:
                    doc_vector.append(model.wv[word])
            if doc_vector:
                document_vector = sum(doc_vector) / len(doc_vector)
                document_vectors.append(document_vector)

        return document_vectors

    def recommend_new_user_content_based(user_cosine_sim):
        hotel_indices = np.argsort(user_cosine_sim)[::-1]

        return hotel_data.loc[hotel_indices[0]]

if st.button('Gợi ý khách sạn'):
    if user_input == '':
        if ((len(selected_facilities) == 0) | (len(selected_around) == 0) | (len(selected_around2) == 0)):
            st.write('Bạn phải chọn ít nhất một giá trị cho các mục trên.')
        else:
            hotel_facilities_vectors = np.array(extract_document_vectors(hotel_fac_corpus))
            hotel_around_vectors = np.array(extract_document_vectors(hotel_ar_corpus))
            hotel_vicinity_vectors = np.array(extract_document_vectors(hotel_vici_corpus))
            hotel_price_vectors = np.array(extract_document_vectors(hotel_pri_corpus))
            user_fac_vec = np.array(extract_document_vectors(user_fac))
            user_ar_vec = np.array(extract_document_vectors(user_ar))
            user_vici_vec = np.array(extract_document_vectors(user_vici))
            user_pri_vec = np.array(extract_document_vectors(user_pri))

            hotel_vec = 0.25*hotel_facilities_vectors + 0.25*hotel_around_vectors + 0.25*hotel_vicinity_vectors + 0.25*hotel_price_vectors
            user_vec = 0.25*user_fac_vec + 0.25*user_ar_vec + 0.25*user_vici_vec + 0.25*user_pri_vec

            sim_synthesis = cosine_similarity(user_vec,hotel_vec)
            # Gọi hàm để gợi ý khách sạn dựa trên thông tin người dùng nhập vào
            recommended_hotels = recommend_new_user_content_based(sim_synthesis)
            df_show = recommended_hotels[['NameHotel','Location','Price','Quality','RatingHotel','CountRating','DistanceCenter','Link']]
            
            df_show = df_show[(df_show.Price >= sidebar_min_price) & (df_show.Price <= sidebar_max_price)]
            df_show = df_show[df_show.Location.isin(sidebar_selected_location)]
            df_show = df_show[(df_show.Quality >= sidebar_min_star) & (df_show.Quality <= sidebar_max_star)]
            df_show = df_show[(df_show.RatingHotel >= sidebar_min_Rating) & (df_show.RatingHotel <= sidebar_max_Rating)]
            df_show = df_show[(df_show.CountRating >= sidebar_min_CountRating) & (df_show.CountRating <= sidebar_max_CountRating)]
            df_show = df_show[(df_show.DistanceCenter >= sidebar_min_distance/1000) & (df_show.DistanceCenter <= sidebar_max_distance/1000)]

            if df_show.shape[0] == 0: st.write('Filter của bạn không nằm trong phạm vi dữ liệu đề xuất của chúng tôi. Mời bạn chỉnh lại filter khác!')
            else:
                def combine_info(row):
                    return f"   <span style='color:blue'><strong>Name Hotel:</strong></span> {row['NameHotel']}  <br>\
                                <span style='color:red'><strong>Location:</strong></span> {row['Location']}  <br>\
                                <span style='color:brown'><strong>Price:</strong></span> {'{:,}'.format(row['Price'])} VND <br>\
                                <span style='color:green'><strong>Quality:</strong></span> {row['Quality']} star &emsp;\
                                <span style='color:green'><strong>Rating:</strong></span> {row['RatingHotel']} point &emsp;\
                                <span style='color:green'><strong>Number of reviews:</strong></span> {row['CountRating']} reivew &emsp;\
                                <span style='color:green'><strong>Distance Center:</strong></span> {row['DistanceCenter']} km\
                                <span><strong>Link detail: </strong></span><a href='{row['Link']}' style='color:sky'>{row['Link']}</a>"
                df_show['Top Recommender'] = recommended_hotels.apply(combine_info, axis=1)
                # st.write(df_show[:10])

                # Lấy HTML từ DataFrame
                html_table = df_show[['Top Recommender']][:10].to_html(escape=False)

                styled_table = df_show[['Top Recommender']][:10].to_html(escape=False, index=False)
                styled_table = f"<div style='background-color: rgb(250, 250, 250); padding: 10px; border-radius: 10px'>{styled_table}</div>"

                st.write(styled_table, unsafe_allow_html=True)
    else:
        user_data = pd.read_csv('data_user.csv')
        if ((user_input not in user_data.IDuser.astype(str).values) | (user_password != '123')):
            st.write('Bạn đã nhập sai tài khoản hoặc mật khẩu, xin vui lòng kiểm tra lại!')
            st.write('Nếu bạn muốn dùng phiên bản cho người mới bắt đầu, hãy để chống ô Id User.')
        else:
            user_input = int(user_input)
            st.write(f'Thông tin tài khoản của bạn:')
            st.write(f'<strong>ID user: </strong>{user_input}', unsafe_allow_html=True)
            st.write(f'<strong>Tên user: </strong>{user_data[user_data.IDuser == user_input].CustomerName.values[0]}', unsafe_allow_html=True)
            
            history_data = pd.read_csv('data_history.csv')
            id_hotel = pd.read_csv('data_hotel.csv')
            st.write(f'<strong>Lịch sử của user:</strong>', unsafe_allow_html=True)
            merged_data = pd.merge(history_data, id_hotel, on='IDhotel')
            st.write(merged_data[merged_data.IDuser == user_input][['NameHotel', 'Location', 'Rating', 'Date']])

            hotel_info = pd.read_csv('data_info_hotel_new.csv')
            rbm_TopK25 = pd.read_csv('rbm_TopK25.csv')
            merged_hotel = pd.merge(hotel_info, id_hotel, on=['NameHotel','Location'])
            recommended_hotels = pd.merge(rbm_TopK25[rbm_TopK25.userID == user_input], merged_hotel, left_on='hotelID', right_on='IDhotel')

            df_show = recommended_hotels[['NameHotel','Location','Price','Quality','RatingHotel','CountRating','DistanceCenter','Link']]
            
            df_show = df_show[(df_show.Price >= sidebar_min_price) & (df_show.Price <= sidebar_max_price)]
            df_show = df_show[df_show.Location.isin(sidebar_selected_location)]
            df_show = df_show[(df_show.Quality >= sidebar_min_star) & (df_show.Quality <= sidebar_max_star)]
            df_show = df_show[(df_show.RatingHotel >= sidebar_min_Rating) & (df_show.RatingHotel <= sidebar_max_Rating)]
            df_show = df_show[(df_show.CountRating >= sidebar_min_CountRating) & (df_show.CountRating <= sidebar_max_CountRating)]
            df_show = df_show[(df_show.DistanceCenter >= sidebar_min_distance/1000) & (df_show.DistanceCenter <= sidebar_max_distance/1000)]

            if df_show.shape[0] == 0: st.write('Filter của bạn không nằm trong phạm vi dữ liệu đề xuất của chúng tôi. Mời bạn chỉnh lại filter khác!')
            else:
                def combine_info(row):
                    return f"   <span style='color:blue'><strong>Name Hotel:</strong></span> {row['NameHotel']}  <br>\
                                <span style='color:red'><strong>Location:</strong></span> {row['Location']}  <br>\
                                <span style='color:brown'><strong>Price:</strong></span> {'{:,}'.format(row['Price'])} VND <br>\
                                <span style='color:green'><strong>Quality:</strong></span> {row['Quality']} star &emsp;\
                                <span style='color:green'><strong>Rating:</strong></span> {row['RatingHotel']} point &emsp;\
                                <span style='color:green'><strong>Number of reviews:</strong></span> {row['CountRating']} reivew &emsp;\
                                <span style='color:green'><strong>Distance Center:</strong></span> {row['DistanceCenter']} km\
                                <span><strong>Link detail: </strong></span><a href='{row['Link']}' style='color:sky'>{row['Link']}</a>"
                df_show['Top Recommender'] = recommended_hotels.apply(combine_info, axis=1)
                # st.write(df_show[:10])

                # Lấy HTML từ DataFrame
                html_table = df_show[['Top Recommender']][:10].to_html(escape=False)

                styled_table = df_show[['Top Recommender']][:10].to_html(escape=False, index=False)
                styled_table = f"<div style='background-color: rgb(250, 250, 250); padding: 10px; border-radius: 10px'>{styled_table}</div>"

                st.write(styled_table, unsafe_allow_html=True)

if st.button('Hybrid-based (cần có tài khoản)'):
    if user_input == '':
        st.write('Cần đang nhập tài khoản để có thể dùng tính năng này!!!')
    elif user_input != '':    
        user_data = pd.read_csv('data_user.csv')
        if ((user_input not in user_data.IDuser.astype(str).values) | (user_password != '123')):
            st.write('Bạn đã nhập sai tài khoản hoặc mật khẩu, xin vui lòng kiểm tra lại!')
        elif ((len(selected_facilities) == 0) | (len(selected_around) == 0) | (len(selected_around2) == 0)):
            st.write('Bạn phải chọn ít nhất một giá trị cho các mục "Bạn chưa có tài khoản".')
        else:
            user_input = int(user_input)
            st.write(f'Thông tin tài khoản của bạn:')
            st.write(f'<strong>ID user: </strong>{user_input}', unsafe_allow_html=True)
            st.write(f'<strong>Tên user: </strong>{user_data[user_data.IDuser == user_input].CustomerName.values[0]}', unsafe_allow_html=True)
            
            history_data = pd.read_csv('data_history.csv')
            id_hotel = pd.read_csv('data_hotel.csv')
            st.write(f'<strong>Lịch sử của user:</strong>', unsafe_allow_html=True)
            merged_data = pd.merge(history_data, id_hotel, on='IDhotel')
            st.write(merged_data[merged_data.IDuser == user_input][['NameHotel', 'Location', 'Rating', 'Date']])

            hotel_info = pd.read_csv('data_info_hotel_new.csv')
            rbm_TopK25 = pd.read_csv('rbm_TopK25.csv')
            merged_hotel = pd.merge(hotel_info, id_hotel, on=['NameHotel','Location'])
            recommended_hotels = pd.merge(rbm_TopK25[rbm_TopK25.userID == user_input], merged_hotel, left_on='hotelID', right_on='IDhotel')
            df_show_CF = recommended_hotels[['NameHotel','Location','Price','Quality','RatingHotel','CountRating','DistanceCenter','Link']]


            hotel_facilities_vectors = np.array(extract_document_vectors(hotel_fac_corpus))
            hotel_around_vectors = np.array(extract_document_vectors(hotel_ar_corpus))
            hotel_vicinity_vectors = np.array(extract_document_vectors(hotel_vici_corpus))
            hotel_price_vectors = np.array(extract_document_vectors(hotel_pri_corpus))
            user_fac_vec = np.array(extract_document_vectors(user_fac))
            user_ar_vec = np.array(extract_document_vectors(user_ar))
            user_vici_vec = np.array(extract_document_vectors(user_vici))
            user_pri_vec = np.array(extract_document_vectors(user_pri))

            hotel_vec = 0.25*hotel_facilities_vectors + 0.25*hotel_around_vectors + 0.25*hotel_vicinity_vectors + 0.25*hotel_price_vectors
            user_vec = 0.25*user_fac_vec + 0.25*user_ar_vec + 0.25*user_vici_vec + 0.25*user_pri_vec

            sim_synthesis = cosine_similarity(user_vec,hotel_vec)
            # Gọi hàm để gợi ý khách sạn dựa trên thông tin người dùng nhập vào
            recommended_hotels = recommend_new_user_content_based(sim_synthesis)
            df_show_CB = recommended_hotels[['NameHotel','Location','Price','Quality','RatingHotel','CountRating','DistanceCenter','Link']]


            for i in range(len(df_show_CF)-1):
                df_show_CB.iloc[i+1] = df_show_CF.iloc[i]
            
            df_show = df_show_CB[:len(df_show_CF)-1]

            df_show = df_show[(df_show.Price >= sidebar_min_price) & (df_show.Price <= sidebar_max_price)]
            df_show = df_show[df_show.Location.isin(sidebar_selected_location)]
            df_show = df_show[(df_show.Quality >= sidebar_min_star) & (df_show.Quality <= sidebar_max_star)]
            df_show = df_show[(df_show.RatingHotel >= sidebar_min_Rating) & (df_show.RatingHotel <= sidebar_max_Rating)]
            df_show = df_show[(df_show.CountRating >= sidebar_min_CountRating) & (df_show.CountRating <= sidebar_max_CountRating)]
            df_show = df_show[(df_show.DistanceCenter >= sidebar_min_distance/1000) & (df_show.DistanceCenter <= sidebar_max_distance/1000)]

            if df_show.shape[0] == 0: st.write('Filter của bạn không nằm trong phạm vi dữ liệu đề xuất của chúng tôi. Mời bạn chỉnh lại filter khác!')
            else:
                def combine_info(row):
                    return f"   <span style='color:blue'><strong>Name Hotel:</strong></span> {row['NameHotel']}  <br>\
                                <span style='color:red'><strong>Location:</strong></span> {row['Location']}  <br>\
                                <span style='color:brown'><strong>Price:</strong></span> {'{:,}'.format(row['Price'])} VND <br>\
                                <span style='color:green'><strong>Quality:</strong></span> {row['Quality']} star &emsp;\
                                <span style='color:green'><strong>Rating:</strong></span> {row['RatingHotel']} point &emsp;\
                                <span style='color:green'><strong>Number of reviews:</strong></span> {row['CountRating']} reivew &emsp;\
                                <span style='color:green'><strong>Distance Center:</strong></span> {row['DistanceCenter']} km\
                                <span><strong>Link detail: </strong></span><a href='{row['Link']}' style='color:sky'>{row['Link']}</a>"
                
                df_show['Top Recommender'] = df_show.apply(combine_info, axis=1)
                # st.write(df_show[:10])

                # Lấy HTML từ DataFrame
                html_table = df_show[['Top Recommender']][:10].to_html(escape=False)

                styled_table = df_show[['Top Recommender']][:10].to_html(escape=False, index=False)
                styled_table = f"<div style='background-color: rgb(250, 250, 250); padding: 10px; border-radius: 10px'>{styled_table}</div>"

                st.write(styled_table, unsafe_allow_html=True)