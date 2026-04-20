[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZubtsoVA/course_work/blob/main/run.ipynb?force=True)

# Physics-Informed CNN для уравнения Блэка-Шоулза

Реализация Physics-Informed Convolutional Neural Network (PI-CNN) для решения уравнения Блэка-Шоулза.

## Описание

Проект решает уравнение Блэка-Шоулза для оценки европейских опционов:

```
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```

где:
- `V(S,t)` — стоимость опциона
- `S` — цена базового актива
- `t` — время
- `σ` — волатильность
- `r` — безрисковая ставка

## Структура проекта

```
├── utils.py      # Утилиты: CubicStretching, BSParams, FDOperators, BSLoss
├── model.py      # Архитектура U-Net и класс Trainer
└── main.py       # Визуализация и запуск обучения
```

## Основные компоненты

### 1. **CubicStretching** (`utils.py`)
Адаптивное сгущение сетки вокруг страйка опциона для повышения точности в критических областях.

### 2. **PiCNN_BlackScholes** (`model.py`)
U-Net архитектура с encoder-decoder структурой для предсказания стоимости опциона.

### 3. **BSLoss** (`utils.py`)
Функция потерь, включающая:
- PDE residual во внутренних точках
- Граничные условия (S=0, S=S_max)
- Терминальное условие (payoff при t=0) (время инвертировано)

### 4. **Trainer** (`model.py`)
Класс для обучения модели с поддержкой DirectML/CUDA.

## Запуск

```python
python main.py
```

## Параметры по умолчанию

- **Сетка**: 256×256 (S × t)
- **Эпохи**: 4000
- **Learning rate**: 5e-4
- **Опцион**: Call, K=1000, T=3 
- **Параметры**: r=0.08, σ=0.28, S_max=7000

## Результаты

Скрипт генерирует:
- `pi_cnn_bs_results.png` — графики обучения и сравнение с аналитическим решением
- `pi_cnn_interior_validation.png` — валидация на внутренних точках

## Требования

```
torch
numpy
matplotlib
scipy
torch-directml  # для AMD GPU (опционально)
```

## Текущие проблемы

* Скачки на границах (s=0 s=s_max)
* Долгая сходимость (>4000 эпох)




