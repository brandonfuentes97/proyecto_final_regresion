import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Proyecto:

    def saludar(self):
        print(f"Hola!")

    def mostrarGraficasColumnas(self, df):
        columnas = df.columns.values
        for i in columnas:
            sns.displot(df[i], kde=True)
            plt.show()

    def mostrarGraficasCorrelacion(self, df):
        df_corr = df.corr().iloc[0]
        columnas = df.columns.values
        for i in columnas:
            plt.scatter(df[i], df['SalePrice'])
            plt.title(f"X vs Y - Coeficiente de Correlacion: {df_corr[i]}")
            plt.xlabel(i)
            plt.ylabel('y')
            plt.show()


    def trainLinearRegression(self, x, y, epochs, print_error_each, alpha):
    
        # Crear matriz de características X
        X = np.vstack([x, np.ones(len(x))]).T
        
        # Inicializar parámetros del modelo
        beta = np.zeros(2)
        
        # Inicializar vector de errores
        errors = np.zeros(epochs)

        # Estructura de datos
        estructura_datos = {}
        
        # Entrenar modelo por el número de epochs especificado
        for epoch in range(epochs):
            # Calcular predicción
            y_pred = np.dot(X, beta)
            
            # Calcular error
            error = np.nanmean((y_pred - y)**2)
            
            # Almacenar error
            errors[epoch] = error
            
            # Imprimir error cada print_error_each epochs
            if epoch % print_error_each == 0:
                print(f"Epoch {epoch}: error = {error}")
            
            # Calcular gradientes
            gradiente = np.mean((y_pred - y)[:, np.newaxis] * X, axis=0)
            
            # Actualizar parámetros
            #beta -= alpha * gradiente
            beta = beta - (alpha*gradiente)

            estructura_datos[epoch] = {'y_pred': y_pred, 'gradiente': gradiente, 'beta': beta}
        
        return errors, estructura_datos
    

    def graficasErrores(self, errors):
        plt.plot(range(len(errors)), errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()


    def graficarModelo(self, n, resultados, columna_predictiva, df, column_real = 'SalePrice'): #DF TRAIN
    
        for index in np.arange(n, len(resultados), n):
            mb = resultados[index]['beta']
            
            plt.scatter(df[columna_predictiva], df[column_real])
            plt.plot(df[columna_predictiva], df[columna_predictiva]*mb[0] + mb[1], linestyle='solid', color="g")
            plt.xlabel(columna_predictiva)
            plt.ylabel(column_real)
            plt.title("Iteración "+str(index))
            plt.show()


    def scikitlearnFitModel(self, df, columna_predictiva, column_y = 'SalePrice'): # df train
        x = df[columna_predictiva].values.reshape(-1, 1)
        y =  df[column_y]
        scikitlearn_modelo = LinearRegression().fit(x, y)
        print("score:", scikitlearn_modelo.score(x, y))
        print("coef:", scikitlearn_modelo.coef_)
        print("intercept:", scikitlearn_modelo.intercept_)
        return scikitlearn_modelo
    

    def obtenerModeloManual(self, resultados):
        index = len(resultados)-1
        return resultados[index]['beta']

    def predecirModelo(self, x, beta):
        matriz_auxiliar = np.hstack((x, np.ones(x.shape) ))
        r = np.matmul(matriz_auxiliar, beta)
        return r

    def calcularEstimacion(self, skl_modelo, modelo_manual, x):
        scikitlearn_predict = skl_modelo.predict(x)
        modelo_manual_predict = self.predecirModelo(x, modelo_manual)
        promedio = np.nanmean(np.array([scikitlearn_predict, modelo_manual_predict]), axis=0)
        return (scikitlearn_predict, modelo_manual_predict, promedio)
    
    def graficaBarrasComparacion(self, scikitlearn, modelo_manual, promedio, titulo):
        plt.rcParams["figure.figsize"] = (4, 3)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        titulos = ['scikitlearn', 'modelo_manual', 'promedio']
        error = [np.nanmean(scikitlearn), np.nanmean(modelo_manual), np.nanmean(promedio)]
        ax.bar(titulos,error)
        plt.title(titulo)
        plt.show()