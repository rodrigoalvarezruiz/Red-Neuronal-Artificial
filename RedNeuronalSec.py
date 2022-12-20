#----------- Importamos las librerias ----------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#----------- Declaramos los eventos pasados ----------
entrada = np.array([1, 6, 30, 7, 70, 43, 503, 201, 1005, 99], dtype = float)

resultados = np.array([0.0254, 0.1524, 0.762, 0.1778, 1.778, 1.0922, 12.776, 5.1054, 25.527, 2.514], dtype=float)

#----- Topografia de la red--------
#creamos una capa tipo dense que cada salida de una neurona va a todas las entradas de la otra neurona.
capa1 = tf.keras.layers.Dense(units = 1, input_shape = [1])

modelo = tf.keras.Sequential([capa1])

#------- Asignamos optimizador y metrica de perdida ----------
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.05),
    loss = 'mean_squared_error'
)

print('Entrenando la red')

#---- Entrenamos el modelo --------
entrenamiento = modelo.fit(entrada, resultados, epochs=500, verbose = False)

#----- Guardamos la red -----
modelo.save('RedNeuronal.h5')
modelo.save_weights('pesos.h5')

#-------- Observemos el comportamiento de nuestra red -------
plt.xlabel('Ciclos de entrenamiento')
plt.ylabel('Errores')
plt.plot(entrenamiento.history['loss'])
plt.show()

#-----Verificar que la red se entreno--------------
print('TERMINAMOS')

#----- Prediccion ---------
i = input('Ingresar el valor en pulgadas: ')
i = float(i)

prediccion = modelo.predict([i])
print('El vlaor en metros es: ', str(prediccion))