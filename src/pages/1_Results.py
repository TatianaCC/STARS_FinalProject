import streamlit as st

st.set_page_config(
    page_title="Page 1",
    page_icon="📄",
)

def main():

    # Insertar una imagen desde una URL
    img_url = 'https://i.postimg.cc/zXm6DHwr/imagen-nueva2-1.jpg'  
    # Inserta la URL de tu imagen aquí
    st.image(img_url, use_column_width=False)
    
    st.title("Welcome back!")
    st.write('Your request has been processed. Please write your email and request ID in the fields below in order to download the results.')

    # Campos de entrada para correo electrónico y ID de solicitud
    email = st.text_input('Email address')
    request_id = st.text_input('Request ID')

    # Ruta al archivo ZIP que deseas ofrecer para descargar
    archivo_zip_path = "mi_archivo.zip"
    # Lee el contenido del archivo ZIP
    with open("../data/Streamlit_data/results/85", "rb") as file:
        file_bytes = file.read()
    
    # Verifica si ambos campos tienen datos antes de mostrar el botón
    if email and request_id:
        # Crea el botón de descarga
        st.download_button(
        label="Dowload",
        data=file_bytes,
        file_name="results.zip",
        mime="application/zip"
        )
        
        
            








# Ejecutar la aplicación de Streamlit
if __name__ == "__main__":
    main()