# PIML-EDP
En este repositorio se consolidan todos los experimentos referentes al Trabajo de Grado de Ingeniería Física, en donde se realizan exploraciones y repasos de Métodos Numéricos y Ecuaciones Diferenciales Parciales, Physics Informed Neural Networks, Variational Physics Informed Neural Networks.

# Contenido
### 1. [Configuración de ambiente](https://github.com/jpaguilarc99/PIML-EDP/tree/main#configuraci%C3%B3n-de-ambiente-en-python)
### 2. [Métodos numéricos](https://github.com/jpaguilarc99/PIML-EDP/tree/main#m%C3%A9todos-num%C3%A9ricos)
### 3. [Física Computacional](https://github.com/jpaguilarc99/PIML-EDP/tree/main#f%C3%ADsica-computacional)
### 4. [Physics Informed Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/tree/main#physics-informed-neural-networks)
### 5. [Variational Physics Informed Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/tree/main#variational-informed-neural-networks)

## Configuración de ambiente en Python

Para poder ejecutar los códigos de este repositorio, se debe crear un ambiente utilizando el gestor de ambientes de Python, directamente desde el CLI del sistema operativo. Específicamente en Windows:

```cmd
python -m venv numerical_methods_env
```

Después de crear nuestro ambiente, debemos instalar las librerías necesarias para su funcionamiento, específicadas en el [requirements.txt](https://github.com/jpaguilarc99/PIML-EDP/blob/main/requirements.txt).

Posteriormente, para activar nuestro ambiente, debemos ir a la carpeta raíz en donde se realizó la instalación del environment y ejecutar el siguiente comando:

```cmd
env\folder\Scripts\activate
```

En nuestro caso de ejemplo,

```cmd
environments\numerical_methods_env\Scripts\activate
```

Cuando el ambiente se encuentre activo, podremos confirmarlo cuando en nuestro CLI aparezca la ruta base de modo que:

```cmd
(numerical_methods_env) environments\numerical_methods_env
```

## Métodos numéricos
Consolidación de códigos realizados en cursos para el fortalecimiento de conocimientos en métodos numéricos. Los temas tratados son: 

- [Roots of High Degree Equations](https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/roots_of_high_degree_equations.ipynb)
- [Interpolation and Curve Fitting](https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/interpolation_and_curve_fitting.ipynb)
- [Numerical Differentiation](https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/numerical_differentiation.ipynb)
- [Numerical Integration](https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/numerical_integration.ipynb)
- [Systems of Linear Equations](https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/systems_of_linear_equations.ipynb)
- [Ordinary Differential Equations](https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/ordinary_differential_equations.ipynb)

## Física Computacional

## Physics Informed Neural Networks

Consolidación de códigos experimentales referentes a las *Physics Informed Neural Networks*, se prueban diferentes enfoques y múltiples arquitecturas para observar sus respectivos desempeños. Las redes experimentadas hasta el momento son:

- [Experiment: Physics Informed Transformer]([https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/roots_of_high_degree_equations.ipynb](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN/piml_transformer.py))
- [Physics Informed DeepONet]([https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/interpolation_and_curve_fitting.ipynb](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN/deepONet.ipynb))
- [Physics Informed Neural Network with Symbolic Regression]([https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/numerical_differentiation.ipynb](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN/ED_PINN_SR.ipynb))
- [Operator Learning]([https://github.com/jpaguilarc99/PIML-EDP/blob/main/numerical-methods/numerical_integration.ipynb](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN/ODIL.py)https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN/ODIL.py)

## Variational Informed Neural Networks
