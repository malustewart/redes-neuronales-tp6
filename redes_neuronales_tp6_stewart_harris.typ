
#import "@preview/basic-report:0.3.1": *
#import "@preview/dashy-todo:0.1.2": todo
#import "@preview/subpar:0.2.2"

#set math.equation(numbering: none)

#show: it => basic-report(
  doc-category: [Redes neuronales #emoji.brain],
  doc-title: "Trabajo práctico 6: \nExtensión de modelos lineales",
  author: "María Luz Stewart Harris",
  affiliation: "Instituto Balseiro",
  language: "es",
  compact-mode: true,
  heading-font: "Vollkorn",
  heading-color: black,
  it
)

= Regularización y selección de modelos

== Regresión sin regularizador

Se realizan regresiones de orden 0 a 4 utilizando el 100% de los datos disponibles (@fig:regresion_tr_1_ajuste). 
Se mide el error de la regresión mediante el error cuadrático medio. 
Se observa que al aumentar el orden del polinomio (equivalente a aumentar la complejidad del modelo) decrece el error (@fig:regresion_tr_1_mse). 

Se repiten las regresiones utilizando el 80% de los datos para entrenar y el 20% restante para evaluar el error del modelo (@fig:regresion_tr_0.8).
Al igual que en el caso anterior, el error de entrenamiento disminuye siempre que aumenta el orden del polinomio. Sin embargo, el error en el set de datos de evaluación tiene un mínimo en orden 1, indicando que para ordenes 2, 3 y 4 el modelo tiene _overfitting_.

El error de evaluación es menor al de entrenamiento para los modelos menos complejos (ordenes 0, 1 y 2). También se observan grandes diferencias en el error de entrenamiento y evaluación dependiendo de cómo se divida el set de datos en 80% para entrenamiento y 20% para evaluación. Una posible explicación es que se debe al tamaño reducido del set de datos.

#subpar.grid(
    figure(
        image("figs/fit_tr_1_lambda_0.0.png"),
        caption: [Ajuste.]
    ), <fig:regresion_tr_1_ajuste>,
    figure(
        image("figs/mse_tr_1_lambda_0.0.png"),
        caption: [Error cuadrático medio.]
    ), <fig:regresion_tr_1_mse>,
    columns: (1fr, 1fr),
    caption: [Ajuste y error cuadrático medio para polinomios de orden 0 a 4 entrenando con 100% de datos ($lambda=0$).],
    label: <fig:regresion_tr_1>
)

#subpar.grid(
    figure(
        image("figs/fit_tr_0.8_lambda_0.0.png"),
        caption: [Ajuste.]
    ), <fig:regresion_tr_0.8_ajuste>,
    figure(
        image("figs/mse_tr_0.8_lambda_0.0.png"),
        caption: [Error cuadrático medio.]
    ), <fig:regresion_tr_0.8_mse>,
    columns: (1fr, 1fr),
    caption: [Ajuste y error cuadrático medio para polinomios de orden 0 a 4 entrenando con 80% de datos ($lambda=0$).],
    label: <fig:regresion_tr_0.8>
)

== Regresión con regularizador _ridge_

Se repiten las regresiones agregando un regularizador _ridge_. Se entrena con el mismo subset del 80% de los datos. La función de cálculo de regresión es:

```python
def get_poly_regression(X, Y, deg, λ=0):
    X = np.array([X**i for i in range(deg + 1)])
    w = np.linalg.inv(X@X.T + λ * np.eye(len(X))) @ X @ Y.T
    return w
```
La @fig:mse_heatmap muestra el error en función del orden de la regresión y el coeficiente de penalización de la regresión _ridge_ $lambda$. 
Al igual que en el caso sin regularización, para un $lambda$ fijo, el error de entrenamiento decrece al aumentar el orden del polinomio pero el de entrenamiento tiene un mínimo (ver ejemplos en @fig:ridge_fixed_lambda).
De forma similar, para un orden de polinomio fijo, al disminuir $lambda$#footnote[Disminuir $lambda$ se corresponde con aumentar la complejidad del modelo] diminuye monotónicamente el error de entrenamiento, pero el error de evaluación tiene un mínimo (ver ejemplos en @fig:ridge_fixed_poly).

Por otro lado, se analiza cualitativamente cómo cambia la posición del mínimo en el error de entrenamiento al modificar alguna de las dos variables de la regresión (orden de polinomio y $lambda$).

Para un $lambda$ fijo, el mínimo de error de evaluación se encuentra en un orden de polinomio $m$. Se observa que para $lambda$ más altos, $m$ es más más alto. Una posible interpretación es que reducir la complejidad del modelo aumentando $lambda$ requiere de más complejidad en la variable $m$ para encontrar un mínimo.

Ejemplo (@fig:ridge_fixed_lambda):
- $lambda=0.1 arrow m_"mín err eval"=1$
- $lambda=100000 arrow m_"mín err eval"=2$


#figure(
    image("figs/mse_tr_0.8.png"),
    caption: [Error cuadrático medio para regresión con regularizador _ridge_ en función del orden de la regresión y el coeficiente de penalización $lambda$.]
)<fig:mse_heatmap>


#subpar.grid(
    figure(
        image("figs/fit_tr_0.8_lambda_0.1.png"),
        caption: [Ajuste ($lambda=0.1$).]
    ),
    figure(
        image("figs/mse_tr_0.8_lambda_0.1.png"),
        caption: [Error cuadrático medio ($lambda=0.1$).]
    ),
    figure(
        image("figs/fit_tr_0.8_lambda_100000.0.png"),
        caption: [Ajuste ($lambda=100000$).]
    ),
    figure(
        image("figs/mse_tr_0.8_lambda_100000.0.png"),
        caption: [Error cuadrático medio ($lambda=100000$).]
    ),
    align:top,
    columns: (1fr, 1fr),
    caption: [Ajuste y error cuadrático medio para polinomios de orden 0 a 4 entrenando con 80% de datos.],
    label: <fig:ridge_fixed_lambda>
)

#subpar.grid(
    figure(
        image("figs/fit_tr_0.8_polydeg_0.png"),
        caption: [Ajuste (polinomio de grado 0).]
    ),
    figure(
        image("figs/mse_tr_0.8_polydeg_0.png"),
        caption: [Error cuadrático medio (polinomio de grado 0).]
    ),
    figure(
        image("figs/fit_tr_0.8_polydeg_1.png"),
        caption: [Ajuste (polinomio de grado 1).]
    ),
    figure(
        image("figs/mse_tr_0.8_polydeg_1.png"),
        caption: [Error cuadrático medio (polinomio de grado 1).]
    ),
    figure(
        image("figs/fit_tr_0.8_polydeg_2.png"),
        caption: [Ajuste (polinomio de grado 2).]
    ),
    figure(
        image("figs/mse_tr_0.8_polydeg_2.png"),
        caption: [Error cuadrático medio (polinomio de grado 2).]
    ),
    align:top,
    columns: (1fr, 1fr),
    caption: [Ajuste y error cuadrático medio para diferentes $lambda$ entrenando con 80% de datos.],
    label: <fig:ridge_fixed_poly>
)
= Anexo
El repostorio de código utilizado para simular y graficar se encuentra en: #box[#link( "https://github.com/malustewart/redes-neuronales-tp6" )]
