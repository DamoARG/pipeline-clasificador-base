# Pipeline de entrenamiento, validación y clasificador base

Primera pre-entrega del proyecto integrador de **Data Science III**.

El repositorio implementa un pipeline reproducible de clasificación multiclase
con PyTorch sobre el dataset clásico **Iris**. El objetivo principal es
demostrar la correcta configuración del entorno, el entrenamiento del modelo,
la validación sobre datos no vistos y el registro de métricas por época.

## Estructura del repositorio

```text
pipeline_clasificador_base/
├── data/
│   └── .gitkeep
├── outputs/
│   ├── accuracy_curve.png
│   ├── iris_mlp.pt
│   ├── loss_curve.png
│   └── metrics.csv
├── src/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

Se utiliza el dataset **Iris**, que contiene 150 observaciones de flores
clasificadas en tres especies:

- Iris setosa
- Iris versicolor
- Iris virginica

Cada observación tiene cuatro variables numéricas relacionadas con el largo y
el ancho de sépalos y pétalos.

El conjunto se divide en:

- 80% para entrenamiento.
- 20% para validación.

La división utiliza `stratify=y` para mantener la proporción de las tres clases.
Las variables se estandarizan con `StandardScaler`, ajustado únicamente sobre
los datos de entrenamiento para evitar filtración de información.

## Arquitectura del modelo

Se implementa un perceptrón multicapa pequeño:

```text
Entrada de 4 variables
        ↓
Capa lineal: 4 → 16
        ↓
ReLU
        ↓
Capa lineal: 16 → 8
        ↓
ReLU
        ↓
Capa lineal: 8 → 3 clases
```

La última capa entrega logits. Por ese motivo, se utiliza
`nn.CrossEntropyLoss`, que internamente aplica la operación necesaria para
clasificación multiclase.

## Configuración del entorno

El script detecta automáticamente el mejor dispositivo disponible:

1. CUDA, si existe una GPU NVIDIA compatible.
2. MPS, en equipos Apple Silicon compatibles.
3. CPU, como alternativa universal.

También se fijan semillas de aleatoriedad para mejorar la reproducibilidad.

## Hiperparámetros

- Optimizador: Adam
- Learning rate: `0.01`
- Épocas: `100`
- Batch size: `16`
- Función de pérdida: `CrossEntropyLoss`
- Semilla: `42`

El learning rate de `0.01` permitió una convergencia rápida y estable para este
dataset pequeño. En problemas más complejos sería conveniente comparar este
valor con tasas menores.

## Ciclo de entrenamiento

Cada batch ejecuta explícitamente:

1. `optimizer.zero_grad()`
2. Forward pass
3. Cálculo de la pérdida
4. `loss.backward()`
5. `optimizer.step()`

Esto evita la acumulación involuntaria de gradientes y utiliza `torch.autograd`
para calcular las derivadas necesarias.

## Validación

Después de cada época, el modelo se evalúa sobre el conjunto de validación.

Durante esta etapa se utilizan:

- `model.eval()`
- `torch.no_grad()`

De esta forma no se calculan gradientes ni se actualizan los pesos.

Las métricas registradas en cada época son:

- Loss de entrenamiento
- Accuracy de entrenamiento
- Loss de validación
- Accuracy de validación

## Interpretación de los resultados

La pérdida de entrenamiento disminuyó desde aproximadamente `1.03` hasta
`0.03`, lo que indica que el modelo aprendió patrones útiles del dataset. En la
ejecución de referencia, la pérdida final de validación fue `0.0903` y la
accuracy de validación alcanzó `96.67%`.

La cercanía entre las métricas de entrenamiento y validación sugiere que el
modelo generaliza correctamente. Si la pérdida de entrenamiento continuara
bajando mientras la pérdida de validación aumentara, sería una señal de
sobreajuste.

Los resultados exactos pueden variar levemente según el hardware y la versión
de las librerías, aunque la fijación de semillas reduce esa variabilidad.

## Ejecución

### 1. Crear un entorno virtual

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el entrenamiento

```bash
python src/train.py
```

Al finalizar se generan:

- `outputs/metrics.csv`
- `outputs/loss_curve.png`
- `outputs/accuracy_curve.png`
- `outputs/iris_mlp.pt`

## Versión de PyTorch

El script imprime la versión instalada al comenzar la ejecución. El proyecto fue ejecutado y verificado con PyTorch `2.10.0+cpu`. También debería funcionar con otras versiones modernas de PyTorch 2.x.

## Conclusión

El pipeline cumple con los requisitos del checkpoint:

- Selección automática de dispositivo.
- Semillas para reproducibilidad.
- Arquitectura `nn.Module`.
- Entrenamiento con Adam.
- Uso correcto de `zero_grad()`, `backward()` y `step()`.
- Validación separada.
- Registro de pérdida y accuracy por época.
- Exportación de métricas, gráficos y pesos del modelo.
