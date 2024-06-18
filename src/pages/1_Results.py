import streamlit as st

# Ruta absoluta al directorio que contiene los resultados
path_results = "C:/Users/milser/Documents/Trasteo_4geeks/STARS_FinalProject/data/Streamlit_data/results/"

st.set_page_config(
    page_title="Resultados",
    page_icon="📄",
)

def main():

    # Insertar una imagen desde una URL
    img_url = 'https://i.postimg.cc/zXm6DHwr/imagen-nueva2-1.jpg'  
    # Inserta la URL de tu imagen aquí
    st.image(img_url, use_column_width=False)
    
    st.title("Welcome back!")
    st.write('Please write your Name and request ID in the fields below in order to check if your request has been processed and download the results.')

    # Campos de entrada para correo electrónico y ID de solicitud
    Name: str = st.text_input('Name:')
    request_id: str = st.text_input('Request ID:')

    # Ruta al archivo ZIP que deseas ofrecer para descargar
    archivo_zip_path = path_results + request_id+"/"+request_id+"_"+Name+".zip"  # Asegúrate de que el archivo ZIP exista en esta ruta
    
    # Verifica si ambos campos tienen datos antes de mostrar el botón
    if Name and request_id:
        try:
            # Lee el contenido del archivo ZIP
            with open(archivo_zip_path, "rb") as file:
                file_bytes = file.read()
            
            # Crea el botón de descarga
            st.download_button(
                label="Download",
                data=file_bytes,
                file_name=request_id+"_"+Name+".zip",
                mime="application/zip"
            )
        except FileNotFoundError:
            st.error("Your data has not been processed yet.")
        except PermissionError:
            st.error("Oops, we do not have any request with this ID.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Ejecutar la aplicación de Streamlit
if __name__ == "__main__":
    main()
