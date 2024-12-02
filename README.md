# CommentAggregator

## Назначение программы

Программа `compare_results.py` предназначена для обработки данных из конфигурационного файла в формате YAML. Она вычисляет средние значения для наборов числовых данных, объединяет текстовые комментарии и визуализирует результаты. Программа может быть полезна для анализа данных, полученных из различных источников, и для сравнения результатов между разными наборами данных.

## Использование

### Подготовка конфигурационного файла

Создайте файл конфигурации в формате YAML, например `config.yaml`, с содержимым, подобным следующему:

```yaml
   datasets:
     dataset1:
       numeric_data: [1, 2, 3]
       text_data: ["Good", "Nice"]
     dataset2:
       numeric_data: [4, 5, 6]
       text_data: ["Excellent"]
   ```

### Запуск программы

Запустите программу с помощью следующей команды:

```bash
python compare_results.py config.yaml --save results.txt
```

- `config.yaml` — путь к вашему конфигурационному файлу.
- `--save results.txt` — (опционально) путь к файлу, в который будут сохранены результаты обработки.

### Пример использования

После выполнения программы вы получите результаты в консоли и в файле `results.txt`. Также будет отображена визуализация распределения средних значений и сравнение результатов для разных наборов данных.

### Тестирование

Для запуска тестов используйте следующую команду:

```bash
python -m unittest test_compare_results.py
```

## Лицензия

Этот проект лицензирован под MIT License. См. файл [LICENSE](LICENSE) для получения дополнительной информации.

## Контакты

Если у вас есть вопросы или предложения, вы можете связаться с автором по электронной почте: mortiss@yandex.ru.
