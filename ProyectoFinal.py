# El programa debe reconocer entre un correo SPAM y otro que no lo es. Se utilizará una bolsa de palabras el cual es un vector que se cuenta cuántas veces se repiten las palabras en un texto. Una bolsa de palabras puede recibir un diccionario.
# Cada correo se vectoriza. Con los correos se creará el dataset. Los vectores serán grandes porque pueden haber muchas palabras diferentes.
# Hay dos zips: uno con correos de SPAM y otro con correos normales. Con esos datos el programa se entrenará.
# Dependiendo de qué tan preciso se clasifiquen los correos, se establecerá la calificación.

# Bolsa de palabras:
# Se deben tener los datos,
# Se escoge el modelo,
# Inicia la fase de desarrollo (se definen los parámetros y se hacen las pruebas pertinentes),
# Producción (filtro de SPAM completado),
# Mantenimiento (se hacen tareas de actualización de la aplicación para adecuarlo a las necesidades actuales)

# Es recomendable que los valores de entrada estén entre 0 y 1.
#
# Saturación de pesos: si conforme ocurre el entrenamiento los valores de los pesos aumentan demasiado. Para solucionar esto se usa la normalización de los datos de entrada. Se intenta tener todo en un rango fijo.
#
# Dos tipos de bolsas de palabras: una en donde verifica que la palabra se encuentra en el correo y otra en donde se contabiliza la presencia de la palabra. No es muy recomendable realizar la contabilización de palabras, por eso se usa un diccionario.
#
# Los correos están en inglés.
#
# Mientras más neuronas se tengan, más probable es caer en un sobreajuste.
#
# Un subajuste es cuando no se clasifican bien todos los datos.
#
# Antes de entregar un proyecto se deben hacer diferentes pruebas con diferentes datos, además de los datos de desarrollo
#
# El proyecto se entrega como una práctica normal. El reporte se sube a la asignación.
#
# Debe mostrar cuántos correos fueron Spam y cuántos no.

import matplotlib as plt




