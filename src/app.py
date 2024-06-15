import streamlit as st

# Insertar una imagen desde una URL
img_url = 'https://i.postimg.cc/zXm6DHwr/imagen-nueva2-1.jpg'  # Inserta la URL de tu imagen aquí
st.image(img_url, use_column_width=False)

# "with" notation
with st.sidebar:
    st.image("https://i.postimg.cc/ZnNzV13B/logo2.jpg", use_column_width=True)
    
    st.title('Stellar')
    st.title('Association')
    st.title('Recognition')
    st.title('System')
    
# Contenido de la aplicación
st.subheader('Discover structures in your star collection')
st.write('')
st.write('Download the following CSV file and fill in the variables required for each star.')  

# Proporcionar enlace de descarga para el archivo CSV de ejemplo
st.markdown(
    """
    [Download CSV Template](https://drive.google.com/uc?export=download&id=15-RFpIMC1aBhnRPuQKnUpc3vccP8LZ3Q)
    """, 
    unsafe_allow_html=True
)
st. write('')

if st.button('Important'):
    st.info('Please, make sure your categorical variables have been factorized and your data has been normalized and does not contain NaN or INF values. We recommend that you remove or deal with the outliers before submitting back the file.')

# Widget para subir el archivo CSV
uploaded_file = st.file_uploader("Please upload your CVS file", type="csv")

if uploaded_file is not None:
    # Mostrar el nombre del archivo
    st.write(f"Filename: {uploaded_file.name}")
    # Mostrar un mensaje de confirmación
    st.success('Well done!') 
    st.write('Since data processing can be compute intensive, we do batch processing.') 
    st.write('Once your request has been completed, we will get back to you, so please provide us with your email address.')
    
    # Collect email address
    email = st.text_input('Email address')

else:
    pass





