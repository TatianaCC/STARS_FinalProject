# Pre_ProyectoFinal
Anotación de ideas y búsquedas previas para el proyecto final

1. Identificación de estructuras estelares con un k-means:
   - Los datos se encuentran fácil en bases de datos de satelites
   - Con características de las estrellas (edad, composición, velocidad, etc) se pueden agrupar en clusters
   - Esto, junto a la posición, permite identificar estructuras de estrellas
2. Identificación de tipos de galaxias con ANN:
   - A partir de fotos de galaxias, identificar de qué tipo son (espirales, elípticas, etc)
   - Las imágenes son facilmente recopilables
3. Clima y precios agrícolas con un modelo de regresión
   - A partir de datos de precipitaciones y temperaturas y del precio de vegetales hacer un predictor de precios de vegetales en función del clima
   - Los datos climatologicos deben estar en la web de la AEMET
4. Traductor de lenguaje de signos con redes neuronales
   - Que al introducir una serie de imagenes de lengua de signos lo traduca a escrito y al contrario
5. Identificador de cáncer de mama con redes neuronales
   - Hay bases de datos de imagenes
   - A partir de parámetros como edad, forma de la masa etc, el modelo determina si es o no cáncer
6. Naive Bayes para determinar si una noticia es falsa o no

| Proyecto   | Base de datos | Enfoque|
|------------|---------------|--------------|
| Identificación de estructuras estelares | VizieR | Clustering |
| Clasificación morfológica de galaxias | Galaxy Zoo| ANN    |
| Traductor de lenguaje de signos | Dactilología del Lenguaje de Signos Español (LSE)   | ANN|
| Identificador de bulos| Kaggle | Naive-Bayes|
| Predicción de cosechas en base a clima | data.world y Meteosat/TuTiempo/NASA | Serie temporal |



  * https://docs.astropy.org/en/stable/io/fits/
