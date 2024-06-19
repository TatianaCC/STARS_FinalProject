# Proyecto Final: Stellar Association Recognition System (STARS)
![Figura 1: Estructuras.](./structures.png)

Dentro de las galaxias las estrellas no están distribuidas de manera uniforme, sino que se agrupan en estructuras de mayor o menor escala. Estudiar estas estructuras nos ayuda a comprender cómo evolucionan las galaxias y esto nos aporta datos cruciales para comprender la materia oscura, la formación estelar, la expansión del universo, su forma, origen y evolución futura. Sin embargo, encontrar estas estructuras, no es nada sencillo. Las estrellas se forman a partir de inmensas nubes de gas y, normalmente, se forman muchas estrellas hermanas a la vez que forman una estructura llamada cúmulo. Con el paso del tiempo y las interacciones entre las estrellas, los cúmulos se disgregan y deforman, haciendo que sea muy difícil reconocer las estrellas que formaban parte de la estructura inicialmente. 

Aun estando el cúmulo bien agrupado, la identificación de las estrellas que pertenecen o no a la estructura en estudio es muy compleja, necesitándose un estudio en profundidad del área del cielo en cuestión mediante diversas técnicas que, al final, son tan subjetivas que raramente producen una buena clasificación de los datos. 

Los efectos de perspectiva, la gran dificultad para medir distancias, el polvo y gas interestelar o no poder medir velocidades son sólo algunos de los problemas. Una mala identificación de las estructuras resulta en conclusiones erróneas de las investigaciones.

En nuestro proyecto hemos querido abordar este problema y usar machine learning para identificar estructuras estelares en el interior de las galaxias con mayor precisión y de una forma más eficiente.

## Objetivo
Este proyecto tiene como objetivo, a partir de la introducción de una lista de estrellas, encontrar agrupaciones de las mismas en base a sus características astrofísicas (tales como edad, movimiento propio, coordenadas espaciales, tipo espectral...) y clasificar las estrellas dadas en las diferentes estructuras encontradas. Para ello se divide el problema en dos partes:
1. Problema de clustering (no supervisado): construir un modelo de clustering que agrupe apropiadamente las estrellas en base a las características más relevantes.
2. Problema de clasificación (supervisado): construir un modelo de clasificación que valide los resultados del clustering.

## Retos
1. Alta dimensionalidad.
2. Limitación de capacidad de computación.
3. Pruebas limitadas debido al tiempo de ejecución, lo que afecta directamente a la optimización.
4. Combinación del modelo no supervisado y modelo supervisado.

## 1. Obtención de datos
La mejor fuente de datos de estrellas de nuestra galaxia es la base de datos de la misión Gaia, que está actualmente en su versión DR3. Esta misión mapea nuestra galaxia y obtiene características de cada estrella individual. En algunos casos estos datos serán más factibles de obtener y en otros casos la estrella está en un entorno de polvo o luz que no permite que se tome la medida.

Como hay más de tres millones de datos en esta fuente y para construir apropiadamente nuestro modelo necesitamos estar seguros de que escogemos un subset que contenga estructuras, haremos lo siguiente:
1. Buscaremos artículos científicos centrados en estudiar cúmulos de estrellas. Esto puede hacerse en la base de datos de papers astrofísicos NASA ADS.
2. Los autores alojan de forma gratuita los datasets que los usan para sus investigaciones en la plataforma VizieR. Buscaremos los datasets correspondientes a los papers que nos interesen y descargamos de ellos 5 columnas: 4 coordenadas de la estrella y el nombre del cúmulo al que los autores creen que pertenece.
3. Unificaremos todos estos datasets en uno sólo.
> Esto está en Clusters.csv
3. Unificaremos los nombres de los cúmulos, ya que el mismo cúmulo puede aparecer repetidamente, pero con nombres diferentes. Para ello:
   1. Haremos una tabla de equivalencia de nombres en los diferentes catálogos.
   > Esta tabla es NamesCatalogEquivalence.csv
   2. Unificaremos los nombres de nuestras estrellas bajo el mismo catálogo.
   > Esto está recogido en RenamedClusters.csv
4. Cruzaremos nuestras estrellas con la base de datos de Gaia DR3 usando el software TopCat, específico para hacer Astronomical Data Query Language (ADQL)
> Esto está en STARSSample.csv

5. Tambien se contruye una tabla de características generales de cada cúmulo (ClustersInfo) que servirá para el tratamiento de valores nulos y outliers. Se aplica el mismo tratamiento a los nombres de los clusters.
> Esto está en ClustersInfo_clean.csv

> Este trabajo está hecho en BuildData.ipynb

## 2.EDA

Comenzaremos el EDA haciendo una exploración general del dataset para luego pasar a hacer el tratamiento de outliers cúmulo a cúmulo. Nuestra target es la columna 'Clusters', siendo el resto de columnas las predictoras.

### 2.1 Exploración general
En BuildData se encuentra la descripción de las columnas de nuestro dataset. A partir de aquí:

1. Se eliminan las columnas repetidas
2. Se eliminan las predictoras que estamos seguros de que son innecesarias para nuestro objetivo
3. Se hace una limpieza general de datos en base a algunas columnas
4. Se realiza el tratamiento de valores nulos e infinitos mediante
   1. Imputación por el valor que tenemos del cluster como objeto en ClustersInfo_clean.csv
   2. Algunos se pueden calcular en base a otras columnas
   3. Si tiene sentido y no hay otra opción, se imputan por 0.
   4. Se eliminan las filas si no hay otra forma de resolver el nan. No se cree conveniente sustituir por ninguna métrica estadística porque al tener datos de zonas muy diferentes de la galaxia, estas métricas no tendrían sentido.
5. Se hace el tratamiento de outliers (ver 2.3)
6. Se eliminan totalmente los cúmulos que hayan quedado con un número de estrellas muy bajo, insuficiente para un tratamiento posterior.
7. Se factorizan las columnas categóricas y se guardan las tablas de equivalencia
8. Se normaliza el dataset.

> Este trabajo se realizará en BuildData
> El sample resultado de estos cambios se guarda como 'STARSSample_EDA1.csv' y factorizado como 'STARSSample_EDA1_fz.csv'
> Las tablas de equivalencia de las categóricas se guarda como 'Cluster_mappings.csv'

### 2.3 Tratamiento de outliers
Dado que en nuestro dataset tenemos datos de diferentes cúmulos que se encuentran en diferentes lugares de nuestra galaxia, no tiene sentido hacer la búsqueda de outliers teniendo en cuenta todo el dataset en su conjunto. Por ello:
1. Dividiremos el dataset inicial por cúmulos.
2. Estudiaremos los outliers en cada grupo por separado.
3. Rearmaremos el dataset.

#### 2.3.1 Búsqueda de outliers
Los cúmulos son grupos de estrellas que se formaron más o menos a la vez a partir de la misma nube de gas, por lo que las estrellas que los componen compartirán, entre otras características, (ordenados de mayor a menor concordancia):
- Edad: todas tendrán una edad similar.
- Movimientos propios: todas se moverán más o menos de la misma forma alrededor del centro galáctico. Para estudiar esto se usa pmRA y pmDE.
- Metalicidad: ya que todas han nacido a partir de la misma nube, todas tendrán más o menos la misma proporción de cada elemento químico. Sin embargo, esto no tiene por qué cumplirse si se han formado estrellas de masas diferentes que producirán elementos pesados a diferentes ritmos.
- Masa: todas tendrán una masa similar.
- Luminosidad y color: si las estrellas tienen una masa similar, tendrán también una luminosidad y un color similares.
- Posición: todas las estrellas tendrán una posición parecida. Sin embargo, esto es más acertado para cúmulos globulares, ya que las estrellas están muy juntas, que para cúmulos abiertos, donde las estrellas han podido disgregarse mucho.

Es posible que en nuestra selección de estrellas tengamos estrellas que aparentemente pertenecen el grupo del cúmulo, pero que en realidad no son hermanas del resto. Estas estrellas infiltradas aparecerán como outliers en las variables antes descritas y el tratamiento correcto es su eliminación del dataset.

Para este trabajo (repetitivo con cada grupo), se contruye la herramienta 'tool.pm_analysis' que nos permite, de forma rápida y gráfica, identificar outliers. Usaremos los movimientos propios, que se espera que tengan una distribución gaussiana, por lo que se calcula la media y la desviación estándar y se escoge un parámetro k para conseguir un intervalo cuyos límites se calculan como l = mean +- k*std. Dividimos este trabajo y guardamos en el valor de la k escogido para cada cúmulo. Posteriormente se ejecuta esta función en bucle y así limpiamos el dataset de outliers.

![Figura 2: Tratamiento de outliers](./outliers.png)

> Este trabajo está hecho en tools.py,  OutTreatment_Nombre.csv, BuildData.ipynb

### 2.4 Feature selection
Uno de los retos de este proyecto es la alta dimensionalidad con la que se trabaja. Tanto en columnas como en filas el número es muy alto, pero dado que esta es la situación real del problema que pretendemos resolver, hemos querido abordarlo igualmente. Por ello, la selección de características es crítica. 

En un primer momento hicimos una selección mediante K-Best dejando 35 características. Es lo que hemos llamado machine1.

Al comenzar las primeras pruebas vimos que los tiempos de ejecución eran muy altos, por lo que decidimos hacer una selección manual, basándonos en nuestro conocimiento del área, dejando 17 columnas. Es lo que llamamos selección human.

Decidimos comprobar si la calidad de los resultados tenía más que ver con las características escogidas o con el número de características, y para ello hicimos una segunda selección con K-Best. Es lo que llamamos machine2.

## 3. Selección del modelo de clustering
Para aportar una solución a problemas reales y no sólo a una simplificación del problema necesitamos:
1. Un modelo escalable para afrontar la alta dimensionalidad.
2. Preferentemente, un modelo que no necesite introducir el número de clusters, ya que los científicos a priori no lo sabrán.

Decidimos probar diferentes modelos (agglomerative clustering, birch, DBSCAN, gaussianmixture,kmeans,meanshift,optics y HDBSCAN). Algunos los descartamos por problemas con la limitación de memoria y de los restantes priorizamos aquellos que no requiera introducir el número de clusteres. Finalmente se escoge HDBSCAN porque:
1. Puede trabajar con conjuntos muy grandes de datos
2. Puede identificar ruido, es decir, datos que no pertenecen a ningún cluster
3. Puede encontrar clusteres no esféricos
4. No es necesario indicar previamente el número de clusteres a encontrar

## 4. Selección del modelo de clasificación
Una forma habitual de trabajar con modelos no supervisados es comprobar su coherencia con modelos supervisados. Por ello, tras encontrar estructuras con HBSCAN pasaremos los resultados por un modelo de clasificación supervisada. Se escoge Random Forest por los siguientes motivos:
1. Es menos propenso al sobreajuste de los datos y tiene una alta precisión.
2. Puede trabajar con alta dimensionalidad y estimar la relevancia de cada característica
3. No necesita normalmente tanta optimización como otros modelos y es fácil de implementar
4. Es robusto frente al ruido

## 5. Coherencia y optimización
Una vez tengamos los resultados de HDBSCAN, se dividirá el dataset en train y test y se entrena el Random Forest. Para validar el clustering calcularemos la coherencia que hay en test entre ambos modelos. Esperamos encontrar una configuración de HDBSCAN que no pueda dar al menos un 95% de coherencia.

Para alcanzar esta coherencia usaremos nos decantamos por una optimización bayesiana por:
1. Su escalabilidad
2. Es más eficiente que grid_search o random_search porque requiere menos evaluaciones
3. Se enfoca hacia las regiones de hiperparámetros más prometedoras gracias a una evaluación estadística que le permite tener en cuenta más información

La función objetivo de la optimización bayesiana se centra en el valor de la métrica silhouette_score que evalúa la calidad y separación de los clusteres.

### 5.1 Selección del espacio de hiperparámetros
Dado los altos tiempos de ejecución es vital seleccionar un buen espacio de hiperparámetros para la optimización bayesiana. 

HDBSCAN es no supervisado, por lo que no podemos hacer uso de métricas para evaluar su rendimiento. Pero aprovechando que hemos construido nuestro dataset conociendo por adelantado los clústeres que tenemos, hemos ideado una forma de, gráficamente, poder evaluar cómo afectan los hiperparámetros al correcto desempeño del modelo. Esto se hace con la herramienta 'tools.graphic_tester'.

![Figura 3: Evaluación gráfica de la influencia de los hiperparámetros](./grapheval_hdbscan.png)

En estas gráficas se muestran en azul los cúmulos que sabemos que tenemos, usando coordenadas la media de las coordenadas galácticas de las estrellas del cúmulo y, como radio de la burbuja, una ponderación del número de estrellas del cluster. En naranja hacemos los mismo, pero con la distribución de las estrellas en diferentes clusters que hace HDBSCAN.

Haber realizado todas estas pruebas con muchos conjuntos de valores y con los 3 feature selection nos ha permitido acotar el espacio de hiperparámetros y obtener el mejor balance posible entre número de columnas y tiempo de ejecución.


## 6. Ensamble
A continuación, se muestra el diagrama de flujo que sigue STARS_class.py y que cumple las siguientes funciones:
1. Encuentra estructuras con HDBSCAN
2. Valida el clustering con un modelo supervisado de clasificación Random Forest
3. Itera haciendo una optimización bayesiana (centrada en la métrica silhouette_score y la coherencia) de HDBSCAN
4. Se detiene cuando el score ponderado es mayor o igual a 95% o en 100 iteraciones.
5. Exporta los resultados, gráficas, datasets y README con información.

![Figura 4: Diagrama de flujo.](./flow.png)

## 7. Despliegue
El despliegue del modelo no es posible de la forma habitual debido a su tamaño y los tiempos de ejecución, por lo que se ejecutará en batch. Sin embargo, se usará Streamlit para que los usuarios puedan hacer una solicitud:
El usuario 
1. Introducirá su email y el dataset con las estrellas entre las que quiere encontrar estructuras.
2. Encontrará una serie de requisitos y recomendaciones que deberá cumplir su dataset para el correcto desempeño del modelo.
3. Recibirá un identificador que deberá guardar para obtener sus datos procesados.

Una vez el modelo se haya ejecutado, el usuario podrá descargar sus resultados usando el identificador y el email.

## Datasets

1. Clusters: contiene una lista de estrellas con sus coordenadas y el cúmulo al que pertenecen. Se ha construido en el paso 1, puntos 1 y 2.
2. NamesCatalogEquivalence: tabla con el nombre de cda cumulo en diferentes catálogos. Se ha construido en el paso 1, punto 3. Se usa para renombrar los clusters y tenerlos todos siempre nombrados igual.
3. RenamedClusters: es el dataset Clusters habiendo cambiado los nombres de los clusters mediante la tabla NamesCatalogEquivalence. Se ha construido en el paso 1, punto 3.
4. STARTSSample: es nuestro dataset inicial. Es el resultado de añadir todas las características de las estrellas a RenamedClusters mediante la herramienta TopCat. Se ha construido en el paso 1, punto 4.
5. ClustersInfo: es una tabla de características generales de cada cúmulo. Es como tener 'la media' de las características de todas las estrellas que componen cada cumulo. Se construye fuera del repo.
6. ClustersInfo_clean: es el resultado de renombrar los clusters y arreglar el formato de ClustersInfo. Se construye en el paso 1, punto 5. Se usará en el tratamiento de valores nulos y outliers.
7. STARSSample_EDA1: dataset inicial tras pasar por el punto 2.1
8. STARSSample_EDA2: dataset STARSSample_EDA1 tras pasar por el punto 2.3.1
9. STARSSample_norm: dataset STARSSample normalizado


## Apéndice 1: Menciones
- LAMOST candidate members of star clusters (Xiang+, 2015)
- APOGEE stars members of 35 star clusters (Garcia-Dias+, 2019)
- Candidate members of star clusters from LAMOST DR2 (Zhang+, 2015)
- Proper motions in the Hyades (Reid 1992)
- Neighboring open clusters with Gaia DR3 (Qin+, 2023)
- New star clusters in M33 (Bedin+, 2005)
- Globular clusters members with Gaia DR2 (Bustos Fierro+, 2019)
- Stellar rotation in young clusters. II. (Huang+, 2006)
- Stellar rotation for 71 NGC 6811 members (Meibom+, 2011)
- NGC 2808 stellar population spectra (Latour+, 2019)
- Members of the young open cluster IC 2395 (Balog+, 2016)
- Spectroscopic membership for NGC 3532 (Fritzewski+, 2019)
- Dynamics of Globular Cluster M15 (Peterson+ 1989)
- Spitzer h and χ Persei candidate members (Cloutier+, 2014)

## Apéndice 2: Herramientas
  
- Gaia: https://www.cosmos.esa.int/web/gaia/early-data-release-3
- TopCat: https://www.star.bris.ac.uk/~mbt/topcat/
- VizieR: https://vizier.cds.unistra.fr/viz-bin/VizieR
- SimbaD: https://simbad.cds.unistra.fr/simbad/
- AstroPy: https://docs.astropy.org/en/stable/io/fits/
