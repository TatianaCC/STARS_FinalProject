import streamlit as st
import sqlite3
import pandas as pd
import os

from streamlit.runtime.uploaded_file_manager import UploadedFile
import sql_table
from STARS_Class import STARS

# Set the page configuration
st.set_page_config(
    page_title="STARS_Proyect",
    layout="centered",  # Optional: can be "centered" or "wide"
    initial_sidebar_state="auto"  # Optional: can be "auto", "expanded", "collapsed"
)

path_CSV = "C:/Users/milser/Documents/Trasteo_4geeks/STARS_FinalProject/data/Streamlit_data/CSV/"

def insert_model(correo, url_modelo, estado=0) -> int | None:
    conn = sqlite3.connect('stars.db')
    c = conn.cursor()
    # Obtener el último ID de la tabla models
    c.execute('SELECT MAX(id) FROM models')
    last_id = c.fetchone()[0] or 0  # Manejar el caso donde la tabla esté vacía

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

def main():
    
    # Insertar una imagen desde una URL
    img_url = 'https://i.postimg.cc/zXm6DHwr/imagen-nueva2-1.jpg'  # Inserta la URL de tu imagen aquí
    st.image(img_url, use_column_width=False)

    # Contenido de la aplicación
    st.write('## Stellar Association Recognition System')
    st.write('##### Discover structures in your star collection')
    st.write('Download the following CSV file and fill in the variables required for each star.')  

    # Proporcionar enlace de descarga para el archivo CSV de ejemplo
    st.markdown(
        """
        [Download CSV Template](https://drive.google.com/file/d/1jCy07vrcSK52uu0LswZ8CVuGMl1RVj8G/view?usp=drive_link)
        """, 
        unsafe_allow_html=True
    )
    st.write('')

    if st.button('Important', key=1):
        st.info('Please, make sure your categorical variables have been factorized and your data has been normalized and does not contain NaN or INF values. We recommend that you remove or deal with the outliers before submitting back the file.')

    # Widget para subir el archivo CSV
    Name = st.text_input('Name')
    uploaded_file: UploadedFile | None = st.file_uploader("Please upload your CSV file", type="csv")

    if uploaded_file is not None and Name:
        # Inserta el modelo en la base de datos y obtiene el ID antes del procesamiento
        
        if st.button("Submit", key="submit_button"):
            db_id = insert_model(Name, url_modelo=path_CSV)
            csv_path = f"{path_CSV}{db_id}.csv"
            st.success(f"Your request ID will be: {db_id}")# Mostrar el botón de "Submit" solo si hay un archivo subido
            temp_df: pd.DataFrame = pd.read_csv(uploaded_file)
            temp_df.to_csv(csv_path, index=False)
            try:
                STARS(csv_path, db_id, Name)
            except Exception as e:
                st.error(f"Error processing the file: {e}")
        st.write('Since data processing can be compute intensive, we do batch processing. Please, give us some time to process your data.')
        st.write('Once tasks are completed, results will be available in the tab "Results". In order to access them you will need your Name and your request ID.')
        
        
    if uploaded_file is None:
        st.warning("Please upload a CSV file.")


    st.write('')
    st.write('What to expect from STARS?')
    st.write('Once your data has been processed, we will provide you the following files:')
    st.write('- Bubble chart showing the clusters found by the unsupervised model HDBSCAN')
    st.write('- Your CSV file with an extra column including the clusters found by the unsupervised model HDBSCAN')
    st.write('- Your processed data')
    st.write('- A report including: HDBSCAN hyperparameters, clusters size and clusters found by HDBScan, Random Forest hyperparameters, coherence verification and number of iterations.')


if __name__ == "__main__":
    sql_table.init_db()
    main()
