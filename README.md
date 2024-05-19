# Deer Detection and Classification

Этот проект нацелен на обнаружение и классификацию парнокопытных на jpg изображениях с  помощью дообученой MobileNet model. Модель включает код для обучения и пользовательский интерфейс построенный с flet для классификации изображений.
## Table of Contents
- [Installation](#installation)
Тренировка модели - укажите директорию с тренировочным датасетомб директорию валидационного датасета и количество классов в validation_generator, training_generator и classes_count соответственно

Использование - укажите директорию ваших изображений и выполните обработку нажав соответствующую кнопку на интерфейсе

Для начала работы клонируйте репозиторий и установите требуемые библиотеки

## Python версии 3.12.3

```sh
git clone https://github.com/YaSoVuH/hack-ai-case-check-deers.git
pip install -r requirements.txt
cd Hakaton_CLIENT
flet run main.py
