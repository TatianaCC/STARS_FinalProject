import streamlit as st
import sqlite3
import pandas as pd
import io

from streamlit.runtime.uploaded_file_manager import UploadedFile
import sql_table
from STARS_class import STARS

path_csv = "C:/Users/milser/Documents/Trasteo_4geeks/STARS_FinalProject/data/Streamlit_data/CSV/"

def insert_model(correo, url_modelo=path_csv, estado=0) -> int | None:
    conn = sqlite3.connect('stars.db')
    c = conn.cursor()
    # Obtener el último ID de la tabla models
    c.execute('SELECT MAX(id) FROM models')
    last_id = c.fetchone()[0]

    # Calcular el nuevo valor para url_modelo
    new_url_modelo = f"{url_modelo}{last_id + 1}.csv"

    # Actualizar la consulta SQL
    c.execute('''
        INSERT INTO models (correo, url_modelo, estado) 
        VALUES (?, ?, ?)
    ''', (correo, new_url_modelo, estado))
    rtn = c.lastrowid
    conn.commit()
    conn.close()
    return rtn

def get_all_models():
    conn = sqlite3.connect('stars.db')
    c = conn.cursor()
    c.execute('SELECT * FROM models')
    rows = c.fetchall()
    conn.close()
    return rows

def main():
    
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

    if st.button('Important',key=1):
        st.info('Please, make sure your categorical variables have been factorized and your data has been normalized and does not contain NaN or INF values. We recommend that you remove or deal with the outliers before submitting back the file.')


    # Widget para subir el archivo CSV
    email = st.text_input('Email address')
    
    uploaded_file: UploadedFile | None = st.file_uploader("Please upload your CVS file", type="csv")
    if uploaded_file is not None:
        db_id=insert_model(email)
        if st.button("patata"):
            print(str(path_csv)+str(db_id)+'.csv')
            temp_df: pd.DataFrame = pd.read_csv(uploaded_file)
            try:
                temp_df.to_csv(str(path_csv) + str(db_id) + '.csv')
            except PermissionError as e:
                st.error(f"No se pudo guardar el archivo: {e}")
            
            STARS(str(path_csv)+str(db_id)+'.csv',db_id) 
            
        st.success('Well done!') 
        st.write('Since data processing can be compute intensive, we do batch processing.') 
        st.write('Once your request has been completed, we will get back to you, so please provide us with your email address.')
    
        # Collect email address

    else:
        pass

if __name__ == "__main__":
    sql_table.init_db()
    main()




