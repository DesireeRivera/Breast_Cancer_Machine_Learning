#PRACTICA MACHINE LEARNING
#Nombre y Apellidos: Desirée Rivera Rodríguez
#Asignatura: Mineria de datos en biomedicina
#Grado: Ingenieria Biomédica

#Se recomienda limpiar los objetos del workspace.


#PROCESAMIENTO DE DATOS.
#Ponemos la direccion del escritorio donde esta nuestra base de datos
setwd("C:/Users/desir/OneDrive/Documentos/1.Ingenieria biomedica/3º/Mineria/Entrega_Algoritmos")
#Cargamos nuestra base de datos
dat <- read.csv("data.csv")

#EXPLORACIÓN DE DATOS
str(dat)#Vemos la estructura de los datos

#Podemos ver que la columna 33 son NA y la columna 1 es el id que no nos proporciona ninguna información por consiguiente las eliminamos
dat <- dat[,-1]
dat <- dat[,-32]
#Para una mejor visualización de las variables independientes distribuimos en las tres categorias
features_mean <- dat[,2:11]
features_se <- dat[,12:21]
features_worst <- dat[,22:31]
#Vemos los estadisticos descriptivos
#Usaremos summary para ver en especial los cuartiles
summary(features_mean)
#Usaremos describe para ver los demás datos como media, mediana, curtosis, sd, rango o desviacion tipica
describe(features_mean)
summary(features_se)
describe(features_se)
summary(features_worst)
describe(features_worst)

#Examinamos si hay outliers
library(gridExtra)
library(ggplot2)
library(ggpubr)
library(corrplot)
library(psych)
boxplot(dat$radius_mean,main= "Diagrama de Cajas Radio", col= "red")
boxplot(dat$texture_mean,main= "Diagrama de Cajas Textura", col= "red")
boxplot(dat$perimeter_mean,main= "Diagrama de Cajas Perimetro", col= "red")
boxplot(dat$area_mean,main= "Diagrama de Cajas Area", col= "red")
boxplot(dat$smoothness_mean,main= "Diagrama de Cajas Smoothness", col= "red")
boxplot(dat$compactness_mean,main= "Diagrama de Cajas Compactness", col= "red")
boxplot(dat$concavity_mean,main= "Diagrama de Cajas Concavity", col= "red")
boxplot(dat$concave.points_mean,main= "Diagrama de Cajas Concave Points", col= "red")
boxplot(dat$symmetry_mean,main= "Diagrama de Cajas simetria", col= "red")
boxplot(dat$fractal_dimension_mean,main= "Diagrama de Cajas Dimension Fractal", col= "red")

boxplot(dat$radius_se,main= "Diagrama de Cajas Radio SE", col= "red")
boxplot(dat$texture_se,main= "Diagrama de Cajas Textura SE", col= "red")
boxplot(dat$perimeter_se,main= "Diagrama de Cajas Perimetro SE", col= "red")
boxplot(dat$area_se,main= "Diagrama de Cajas Area SE", col= "red")
boxplot(dat$smoothness_se,main= "Diagrama de Cajas Smoothness SE", col= "red")
boxplot(dat$compactness_se,main= "Diagrama de Cajas Compactness SE", col= "red")
boxplot(dat$concavity_se,main= "Diagrama de Cajas Concavity SE", col= "red")
boxplot(dat$concave.points_se,main= "Diagrama de Cajas Concave Points SE", col= "red")
boxplot(dat$symmetry_se,main= "Diagrama de Cajas Simetria SE", col= "red")
boxplot(dat$fractal_dimension_se,main= "Diagrama de Cajas Dimension Fractal SE", col= "red")


boxplot(dat$radius_worst,main= "Diagrama de Cajas Radio Worst", col= "red")
boxplot(dat$texture_worst,main= "Diagrama de Cajas Textura Worst", col= "red")
boxplot(dat$perimeter_worst,main= "Diagrama de Cajas Perimetro Worst", col= "red")
boxplot(dat$area_worst,main= "Diagrama de Cajas Area Worst", col= "red")
boxplot(dat$smoothness_worst,main= "Diagrama de Cajas Smoothness Worst", col= "red")
boxplot(dat$compactness_worst,main= "Diagrama de Cajas Compactness Worst", col= "red")
boxplot(dat$concavity_worst,main= "Diagrama de Cajas Concavity Worst", col= "red")
boxplot(dat$concave.points_worst,main= "Diagrama de Cajas Concave Points Worst", col= "red")
boxplot(dat$symmetry_worst,main= "Diagrama de Cajas Simetria Worst", col= "red")
boxplot(dat$fractal_dimension_worst,main= "Diagrama de Cajas Dimension Fractal Worst", col= "red")


#Miramos la correlaccion de nuestras variables quitando la variable predictora

datos_cancer_correlacion = dat[,-1]
corrplot(cor(datos_cancer_correlacion))

#Vamos a ver los datos estadísticos de nuestra variable a predecir
#Convertimos en factor la variable a predecir
dat$diagnosis <- factor(dat$diagnosis, levels= c("B","M"), labels = c("Benigno","Maligno"))
#Vemos la proporción de los posibles valores de nuestra variable a predecir
table(dat$diagnosis)
prop.table(table(dat$diagnosis))
barplot(table(na.omit(dat$diagnosis)), main= "Diagrama de Barras Diagnóstico", col= "blue")





##################################################################
########################### K-MEANS ############################
##################################################################

#Realizamos las funciones de normalización y tipificación de los datos para medir distancias
nom <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}


tip <- function(x){
  return(x-mean(x)/sd(x))
}

#Normalizamos los datos 

dat_n <- data.frame(lapply(dat[2:31] ,nom))
#Comprobamos que se han normalizado los datos
summary(dat_n$area_mean)

#Preparamos los datos con los conjuntos Train y Test
set.seed(123)
train <- sort(sample(1:nrow(dat_n),(nrow(dat_n)*70)/100))

df_train <- dat_n[train,]
df_test <- dat_n[-train,]

#Guardamos los resultados del diagnostico correspondiente al train y al test anteriormente realizados.
df_train_labels <- dat[train,1]
df_test_labels<- dat[-train,1]

library (class)
#Probamos con un k=21 ya que se recomienda utilizar la raiz cuadrada del numero de observaciones
dat_test_pred <- knn (train = df_train, test = df_test, 
                      cl= df_train_labels, k=21)
head(dat_test_pred )
table(dat_test_pred)
table(df_test_labels)

#Evaluamos el modelo aplicando las medidas de rendimiento
library(caret)
confusionMatrix(df_test_labels,dat_test_pred)
precision(data = df_test_labels, reference = dat_test_pred, relevant = "Benigno")
recall(data = df_test_labels, reference = dat_test_pred, relevant = "Benigno")
F_meas(data = df_test_labels, reference = dat_test_pred, relevant = "Benigno")

#Vamos a crear un vector para ver cuantos k es mejor

for (i in 1:30) {
  dat_test_pred <- knn (train = df_train, test = df_test, 
                        cl= df_train_labels, k=i)
print('Con k igual a ')
print(i)

 print(confusionMatrix(df_test_labels,dat_test_pred))
}
#Haciendo este bucle nos damos cuenta que el mejor valor es k=7 por lo que vamos a ver sus medidas de rendimiento más detalladamente

dat_test_pred <- knn (train = df_train, test = df_test, 
                      cl= df_train_labels, k=7)
confusionMatrix(df_test_labels,dat_test_pred)
precision(data = df_test_labels, reference = dat_test_pred, relevant = "Benigno")
recall(data = df_test_labels, reference = dat_test_pred, relevant = "Benigno")
F_meas(data = df_test_labels, reference = dat_test_pred, relevant = "Benigno")
#Es muy buen modelo pero vamos a intentar mejorar la precisión para ello tipificamos. Repetimos todo el ejercicio pero con los datos tipificados
dat_z <- as.data.frame(lapply(dat[2:31], tip))
dat_train <- dat_z[train, ]
dat_test <- dat_z[-train, ]
dat_train_labels <- dat[train, 1]
dat_test_labels <- dat[-train, 1]
dat_test_pred <- knn(train = dat_train, test = dat_test,
                      cl = dat_train_labels, k=7)

precision(data = dat_test_labels, reference = dat_test_pred, relevant = "Benigno")
recall(data = dat_test_labels, reference = dat_test_pred, relevant = "Benigno")
F_meas(data = dat_test_labels, reference = dat_test_pred, relevant = "Benigno")

confusionMatrix(dat_test_labels,dat_test_pred)
#Vemos que al tipificar hay muy buenos resultados en las medidas de rendimiento pero son un pelin mejor al normalizar



##################################################################
########################### SVM ############################
##################################################################

#Para el SVM, es importante normalizar los datos para que estén todos en la misma escala.
# En este caso no es necesario transformar los datos ya que lo hará el modelo automáticamente.
# Así mismo, sólo trabaja con datos numéricos, pero no es un problema dada la BBDD disponibles.
#Preparamos nuestros datos de entrenamiento y test
set.seed(123)
train <- sort(sample(1:nrow(dat),(nrow(dat)*70)/100))
df_train <- dat[train,]
df_test <- dat[-train,]

library(kernlab)

#Aplica el modelo y a la vez normaliza.
# Búsca de un hiperplano que separe de forma óptima a los puntos de una clase de la de otra, que eventualmente han podido ser previamente proyectados a un espacio de dimensionalidad superior.
# La característica fundamental es buscar el hiperplano que tenga la máxima distancia (margen) con los puntos que estén más cerca de él mismo. De esta forma, los puntos del vector que son etiquetados con una categoría estarán a un lado del hiperplano y los casos que se encuentren en la otra categoría estarán al otro lado

df_classifier <- ksvm(diagnosis ~ ., data = df_train,
                          kernel = "vanilladot")

#Vemos su comportamiento predictor
df_predictions <- predict(df_classifier, df_test)

head(df_predictions)

library(caret)
confusionMatrix(df_test$diagnosis, df_predictions)
precision(data = df_test$diagnosis, reference = df_predictions, relevant = "Benigno")
recall(data = df_test$diagnosis, reference = df_predictions, relevant = "Benigno")
F_meas(data = df_test$diagnosis, reference = df_predictions, relevant = "Benigno")
#Aunque es un gran algoritmo de aprendizaje con una precisión, coeficiente Kappa y Medida F muy buenas, vamos a ver si se puede mejorar cambiando el parametro de Kernel
df_classifier_rbf <- ksvm(diagnosis ~ ., data = df_train,
                              kernel = "rbfdot")

df_predictions_rbf <- predict(df_classifier_rbf,
                                  df_test)

confusionMatrix(df_test$diagnosis, df_predictions_rbf)
precision(data = df_test$diagnosis, reference = df_predictions_rbf, relevant = "Benigno")
recall(data = df_test$diagnosis, reference = df_predictions_rbf, relevant = "Benigno")
F_meas(data = df_test$diagnosis, reference = df_predictions_rbf, relevant = "Benigno")

#Vamos a compararlo cambiando el parametro kernel
df_classifier_t <- ksvm(diagnosis ~ ., data = df_train,
                          kernel = "tanhdot")

df_predictions_t <- predict(df_classifier_t,
                              df_test)

confusionMatrix(df_test$diagnosis, df_predictions_t)
precision(data = df_test$diagnosis, reference = df_predictions_t, relevant = "Benigno")
recall(data = df_test$diagnosis, reference = df_predictions_t, relevant = "Benigno")
F_meas(data = df_test$diagnosis, reference = df_predictions_t, relevant = "Benigno")

#De todos estos podemos ver que usando el parametro de kernel de vanilladot y rbfdot nos da el mejor resultando teniendo la misma precisión y coeficiente de Kappa
#Lo unico que cambia es la sensibilidad y la especifidad por lo que habria que ver que es más conveniente




     ##################################################################
########################### ARBOLES DE DECISIÓN ############################
     ##################################################################
colnames(dat)

table(dat$diagnosis)
#Cogemos nuestra muestra para el entrenamiento y test
set.seed(123)
#Lo que hacemos es reordenar la base aleatoriamente para eliminar posibles sesgos,
#es una alternativa para dividir train y test sin sample.
dat_rand <- dat[order(runif(569)), ]

#comprobamos los mismos valores para asegurar mismos resultados del modelo.
summary(dat_rand$diagnosis)
summary(dat$diagnosis)
#Dividimos las muestras reordenados en 70% para el train y 30% para el test
df_train <- dat_rand[1:398, ]
df_test <- dat_rand[399:569, ]

prop.table(table(df_train$diagnosis))
prop.table(table(df_test$diagnosis))

#Usamos el algoritmo de C50
library(C50)
#La función C5.0 () facilita agregar el boosting a nuestro árbol de decisión. La aplicamos a todas las columnas de la base excepto a la columna predictora
df_model <- C5.0(df_train[-1],df_train$diagnosis,rules=FALSE)
df_model

#Vemos como ha tomado sus decisiones y que variables ha usado
summary(df_model)#Podemos ver que hay un 1.3% de error

df_pred <- predict(df_model,df_test)
#Medidas de rendimiento
library(gmodels)
confusionMatrix(df_test$diagnosis, df_pred)
precision(data = df_test$diagnosis, reference = df_pred, relevant = "Benigno")
recall(data = df_test$diagnosis, reference = df_pred, relevant = "Benigno")
F_meas(data = df_test$diagnosis, reference = df_pred, relevant = "Benigno")

#Utilizamos el concepto de Boosting para mejorar el modelo
#En este agregaremos un parametros que será el número de arboles separados. Estos arboles juntaran su fortalezas y debilidades para mejorar el modelo.
#Probamos con 10 arboles
df_boost10 <- C5.0(df_train[-1],df_train$diagnosis,trials=10)
df_boost10

#Vemos como ha tomado sus decisiones y que variables ha usado
summary(df_boost10)
df_pred10 <- predict(df_boost10,df_test)
confusionMatrix(df_test$diagnosis, df_pred10)
precision(data = df_test$diagnosis, reference = df_pred10, relevant = "Benigno")
recall(data = df_test$diagnosis, reference = df_pred10, relevant = "Benigno")
F_meas(data = df_test$diagnosis, reference = df_pred10, relevant = "Benigno")
#Vemos que mejora respecto al anterior pero vamos a ver si se puede mejorar aun más
df_boost20 <- C5.0(df_train[-1],df_train$diagnosis,trials=20)
df_boost20
#Vemos como ha tomado sus decisiones y que variables ha usado
summary(df_boost20)
df_pred20 <- predict(df_boost20,df_test)
confusionMatrix(df_test$diagnosis, df_pred20)
precision(data = df_test$diagnosis, reference = df_pred20, relevant = "Benigno")
recall(data = df_test$diagnosis, reference = df_pred20, relevant = "Benigno")
F_meas(data = df_test$diagnosis, reference = df_pred20, relevant = "Benigno")
#Mejora. Ahora vamos a ver hasta cuantos arboles de decisión mejora ya que no más número de arboles supone una mejora
df_boost30 <- C5.0(df_train[-1],df_train$diagnosis,trials=30)
df_boost30
#Vemos como ha tomado sus decisiones y que variables ha usado
summary(df_boost30)
df_pred30 <- predict(df_boost30,df_test)
confusionMatrix(df_test$diagnosis, df_pred30)
precision(data = df_test$diagnosis, reference = df_pred30, relevant = "Benigno")
recall(data = df_test$diagnosis, reference = df_pred30, relevant = "Benigno")
F_meas(data = df_test$diagnosis, reference = df_pred30, relevant = "Benigno")
#Como podemos comprobar con 20 trials tenemos muy buen modelo por lo que nos quedamos con ese
#Dibujamos su arbol y asi ver como ha realizado esas decisiones de forma más visual
library(dplyr)
plot(df_boost20)



##################################################################
########################### NAIVES BAYES ############################
##################################################################
#Preparamos nuestras muestras de entrenamiento y test
#dat <- as.data.frame(dat)
set.seed(123)
train <- sort(sample(1:nrow(dat),(nrow(dat)*70)/100))
#Quitamos nuestra variable a predecir
df_train <- dat[train,-1]
df_test <- dat[-train,-1]
#Incorporamos en las siguientes variable y las convertimos en factor
cod_train <- dat[train,1]
cod_test <- dat[-train,1]
cod_train <- as.factor(cod_train)
cod_test <- as.factor(cod_test)
#Comprobamos que no haya un desbalanceo de clases
prop.table(table(cod_test))
prop.table(table(cod_train))


library(e1071)
#Esta función nos pide como argumentos la variable objetivo para clasificar y los datos que serán usados.
dat_classifier <- naiveBayes(df_train, cod_train)
#Predecimos con el test
dat_test_pred <- predict(dat_classifier,df_test)
prop.table(table(dat_test_pred))

#Medidas de rendimiento
library(caret)
confusionMatrix(cod_test, dat_test_pred)
precision(data = cod_test, reference = dat_test_pred, relevant = "Benigno")
recall(data = cod_test, reference = dat_test_pred, relevant = "Benigno")
F_meas(data = cod_test, reference = dat_test_pred, relevant = "Benigno")

##################################################################
########################### K-MEANS ############################
##################################################################

#Establecemos la semilla por los k centroides aleatorios
set.seed(123)
#Escalamos (normalizamos) y asignamos clusters aleatorios
library(stats)
#Aplicamos en modelo sobre toda la base de datos menos la columna predictora. Vamos a probar con k=3 para ver que tenemos
dat.km <- kmeans(scale(dat[,-1]),3,nstart=25)
names(dat.km)
dat.km$cluster#asignacion observaciones a clusters
library("factoextra")
fviz_cluster(dat.km, data = dat[,-1],
             palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

dat.km$withinss#inercia intra grupos
dat.km$tot.withinss#inercia intra total
sum(dat.km$withinss)
#Vemos que el parametro tot.withinss es la que nos interesa para ver la dispersión y por lo tanto determinar el numero de clusters adecuado
#Buscamos cual es el k adecuado
dispersion <- NULL
for (i in 1:20) {
  res <- kmeans(scale(dat[,-1]),i,nstart=25)
  dispersion <- c(dispersion,res$tot.withinss,i)
}

dispersion <- as.data.frame(matrix(dispersion,ncol = 2,byrow = T))
dispersion
colnames(dispersion) <- c("Varianza","Grupos")
#Usamos el metodo del codo 
plot(dispersion$Grupos,dispersion$Varianza, main= "Metodo del codo")
#Vemos que no esta muy claro cuantos k coger por lo que vamos a probar con el método silhouette que es más preciso
fviz_nbclust(dat[,-1],kmeans,method="silhouette")
set(123)
res.km_5 <- kmeans(scale(dat[, -1]), 2, nstart = 25)
table(res.km_5$cluster)
table(dat$diagnosis)
dat$diagnosis
#No utilizamos medidas de rendimiento ya que aqui no tenemos datos etiquetados solo buscamos clusters

#Dibujamos los clusters
fviz_cluster(res.km_5, data = dat[, -1], 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)
#Podemos comprobar que elige k=2 correspondiendose a lo que ya sabemos que hay dos grupos a predecir: Benigno y Maligno

#En el documento adjunto al trabajo se analizarán todas las medidas de rendimiento para ver cual es el más adecuado de todos los algoritmos



