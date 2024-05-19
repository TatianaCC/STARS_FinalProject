# Proyecto Final: Stellar Association Recognition System (STARS)
## Objetivo
Este proyecto tiene como objetivo, a partir de la introducción de una lista de estrellas, encontrar agrupaciones de las mismas en base a sus características astrofísicas (tales como edad, movimiento propio, coordenadas espaciales, tipo espectral...) y, posteriormente se categorizarán estas estructuras en cúmulos abiertos o cerrados. Para ello se divide el problema en dos partes:
1. Problema de clustering: construir un modelo de clustering que agrupe apropiadamente las estrellas en base a las características más relevantes.
2. Problema de RNA: construir un modelo que sea capaz de diferenciar cúmulos abiertos o globulares. (Ver Figura 1)

![Figura 1: tipos de cúmulos.]('./TiposCumulo.jpg')

## 1. Obtención de datos
Los datos podrán buscarse en la base de datos VizieR (https://vizier.cds.unistra.fr/viz-bin/VizieR). Este es un portal donde se alojan de forma gratuita los datasets que los científicos usan para sus investigaciones. Los datos generalmente provienen de satélites y observatorios que hacen observaciones del cielo bucando objetos concretos. Algunas misiones se centran en galaxias, otras en supernovas, etc. En nuestro caso estamos buscando datos de estrellas concretas.

El mejor catálogo de estrellas actualmente es el obtenido por la misión Gaia, que en su última actualización es Gaia DR3. Esta misión mapea nuestra galaxia y obtiene características de cada estrella indicidual. En algunos casos estos datos serán más factibles de obtener y en otros casos la estrella está en un entorno de polvo o luz que no permite que se tome la medida. Actualmente este catálogo cuenta con casi 2millones de estrellas mapeadas. 

Para escoger estrellas que pertenezcan a cúmulos, ya sean abiertos o cerrados, nos fijaremos en trabajos previos de investigadores que hayan hecho una selección aproximada de estrellas para uno o varios cumulos en concreto. 
1. Tomaremos de sus samples de datos las coordenadas de las estrellas y el cúmulo al que pertenecen.
2. Unificaremos todas las estrellas en un únido dataset.
3. Unificaremos los nombres de los cúmulos, ya que puede estas el mismo cumulo repetidamente, con nombres diferentes. Para ello:
   1. Haremos una tabla de equivalencia de nombres en los diferentes catálogos.
   2. Unificaremos los nombres de nuestras estrellas bajo el mismo catálogo.
4. Cruzaremos nuestras estrellas con la base da datos de Gaia DR3 usando el software TopCat, específico para hacer Astronomical Data Query Language (ADQL)

## EDA


  * https://docs.astropy.org/en/stable/io/fits/
