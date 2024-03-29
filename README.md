# PIML-EDP
En este repositorio se consolidan todos los experimentos referentes al Trabajo de Grado de Ingeniería Física, en donde se realizan exploraciones respecto a Métodos Numéricos y Ecuaciones Diferenciales Parciales, Physics Informed Neural Networks, Variational Physics Informed Neural Networks, entre otros.

## Contenido

1. [Configuración de ambiente](https://github.com/jpaguilarc99/PIML-EDP/tree/main#configuraci%C3%B3n-de-ambiente-en-python)
2. [Métodos numéricos](https://github.com/jpaguilarc99/PIML-EDP/tree/main#m%C3%A9todos-num%C3%A9ricos)
3. [Física Computacional](https://github.com/jpaguilarc99/PIML-EDP/tree/main#f%C3%ADsica-computacional)
4. [Physics Informed Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/tree/main#physics-informed-neural-networks)
5. [Poisson Physics Informed Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/tree/main/PoissonPINNs)
6. [Integración numérica](https://github.com/jpaguilarc99/PIML-EDP/tree/main/Numerical_Integration)
7. [Variational Physics Informed Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/tree/main#variational-informed-neural-networks)

## Configuración de ambiente en Python

### VENV

Para poder ejecutar los códigos de este repositorio, se debe crear un ambiente utilizando el gestor de ambientes de Python, directamente desde el CLI del sistema operativo. Específicamente en Windows:

```cmd
python -m venv numerical_methods_env
```

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
Después de crear y activar nuestro ambiente, debemos instalar las librerías necesarias para su funcionamiento, específicadas en el [requirements.txt](https://github.com/jpaguilarc99/PIML-EDP/blob/main/requirements.txt). Ejecutamos el comando de instalación de los paquetes necesarios:

```cmd
(numerical_methods_env) environments\numerical_methods_env\pip install -r requirements.txt
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

En esta sección se consolidan los códigos realizados para resolver problemas físico mediante el uso de métodos numéricos:

- [Física Computacional](https://github.com/jpaguilarc99/PIML-EDP/tree/main/computational_physics)

## Physics Informed Neural Networks

Consolidación de códigos experimentales referentes a las *Physics Informed Neural Networks*, se prueban diferentes enfoques y múltiples arquitecturas para observar sus respectivos desempeños. Las redes experimentadas hasta el momento son:

- [Experiment: Physics Informed Transformer](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN_Experiments/HeatPDE_transformer.py)
- [Experiment: Physics Informed DeepONet](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN_Experiments/PIDeepONet.ipynb)
- [Experiment: Physics Informed Neural Network with Symbolic Regression](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN_Experiments/PINN_Symbolic_Reg.ipynb)
- [Experiment: Harmonic Physics Informed Neural Network](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PINN_Experiments/Harmonic_PINN.ipynb)
- [Experiment: Heat and Burgers Equations](https://github.com/jpaguilarc99/PIML-EDP/tree/main/PINN_Experiments/PINN_FDM)

## Poisson Physics Informed Neural Networks

En esta sección se hace la experimentación de diferentes tipos de arquitecturas de redes neuronales para dar solución a la Ecuación Diferencial Parcial de Poisson utilizando el enfoque de *Physics Informed Neural Networks*. Las arquitecturas empleadas para solucionar el problema de Poisson son:

- [Feed Forward Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PoissonPINNs/Poisson1D_NNvsPINN/Poisson1D_PINN.ipynb)
- [Convolutional Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PoissonPINNs/Poisson1D_CNNvsPICNN/PoissonCNN1D.ipynb)
- [Recurrent Neural Networks](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PoissonPINNs/Poisson1D_RNNvsPIRNN/Poisson1D_RNN.ipynb)
- [Transformer Architecture](https://github.com/jpaguilarc99/PIML-EDP/blob/main/PoissonPINNs/Poisson1D_TransformervsPITN/PoissonTransformer1D.ipynb)

## Integración numérica



## Variational Physics Informed Neural Networks

En desarrollo...

## Licencia

Todo el código está bajo licencia MIT y el contenido bajo licencia
Creative Commons Attribute.

El contenido de este repositorio está bajo
[licencia Creative Commons Attribution 4.0](http://choosealicense.com/licenses/cc-by-4.0/),
y el código que lo acompaña bajo
[licencia MIT](https://opensource.org/licenses/mit-license.php).