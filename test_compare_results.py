import unittest
from pathlib import Path
import logging
import yaml

from compare_results import (
    InputData,
    ProcessingResult,
    load_config,
    compute_average,
    combine_comments,
    process_data,
    clean_text
)


class TestComputeAverage(unittest.TestCase):
    def test_average_with_numbers(self):
        self.assertAlmostEqual(compute_average([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(compute_average([10.5, 2.5, 3.0]), 5.333333333333333)

    def test_average_with_none(self):
        self.assertAlmostEqual(compute_average([1, None, 2, None, 3]), 2.0)

    def test_average_with_empty_list(self):
        self.assertEqual(compute_average([]), 0.0)

    def test_average_with_non_numeric(self):
        with self.assertRaises(TypeError):
            compute_average([1, 'two', 3])

        with self.assertRaises(TypeError):
            compute_average([1, 2, None, 'three'])

    def test_average_with_non_list(self):
        with self.assertRaises(TypeError):
            compute_average("not a list")

        with self.assertRaises(TypeError):
            compute_average(123)

    def test_average_with_large_numbers(self):
        self.assertAlmostEqual(compute_average([1e10, 1e10, 1e10]), 1e10)

    def test_average_with_large_dataset(self):
        large_data = [i for i in range(1, 10001)]  # Список от 1 до 10000
        self.assertAlmostEqual(compute_average(large_data), 5000.5)

    def test_average_with_mixed_types(self):
        self.assertAlmostEqual(compute_average([1, 2, 3]), 2.0)  # Изменено на целые числа

    def test_average_with_single_element(self):
        self.assertAlmostEqual(compute_average([5]), 5.0)  # Один элемент

    def test_average_with_only_none(self):
        self.assertEqual(compute_average([None, None]), 0.0)  # Только None

    def test_average_with_invalid_data(self):
        with self.assertRaises(TypeError):
            compute_average([1, 2, 'three', 4])


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Создание временного файла конфигурации
        self.config_path = Path("test_config.yaml")
        config_content = """
           datasets:
             dataset1:
               numeric_data: [1, 2, 3]
               text_data: ["Comment 1", "Comment 2"]
           """
        with self.config_path.open('w', encoding='utf-8') as f:
            f.write(config_content)

    def tearDown(self):
        # Удаление временного файла конфигурации
        if self.config_path.exists():
            self.config_path.unlink()

    def test_combine_comments(self):
        self.assertEqual(combine_comments(["Hello", "World"]), "Hello | World")
        self.assertEqual(combine_comments([]), "")
        self.assertEqual(combine_comments(["Test", "Comment"]), "Test | Comment")
        self.assertEqual(combine_comments(["Single"]), "Single")  # Один комментарий
        self.assertEqual(combine_comments([None, None]), "")  # Все значения None
        self.assertEqual(combine_comments(["Hello", None, "World"]), "Hello | World")  # Смешанные типы

    def test_load_config(self):
        # Создание временного YAML файла для тестирования
        config_content = """
        datasets:
          dataset1:
            numeric_data: [1, 2, 3]
            text_data: ["Comment 1", "Comment 2"]
          dataset2:
            numeric_data: [4, 5, 6]
            text_data: ["Comment 3"]
        """
        config_path = Path("test_config.yaml")
        with config_path.open('w', encoding='utf-8') as f:
            f.write(config_content)

        input_data = load_config(config_path)
        self.assertIn('dataset1', input_data)
        self.assertEqual(input_data['dataset1'].numeric_data, [1.0, 2.0, 3.0])  # Проверка на float
        self.assertEqual(input_data['dataset1'].text_data, ["Comment 1", "Comment 2"])
        config_path.unlink()

    def test_load_config_with_missing_keys(self):
        config_content = """
        datasets:
          dataset1:
            numeric_data: [1, 2, 3]
        """
        config_path = Path("test_config_missing_keys.yaml")
        with config_path.open('w', encoding='utf-8') as f:
            f.write(config_content)

        input_data = load_config(config_path)
        self.assertIn('dataset1', input_data)
        self.assertEqual(input_data['dataset1'].text_data, [])  # Проверка на пустой список
        config_path.unlink()

    def test_load_config_with_invalid_yaml(self):
        malformed_config_path = Path("malformed_config.yaml")
        with malformed_config_path.open('w', encoding='utf-8') as f:
            f.write("not a valid yaml")
        with self.assertRaises(ValueError):
            load_config(malformed_config_path)
        malformed_config_path.unlink()

    def test_clean_text(self):
        self.assertEqual(clean_text("Hello\x00World"), "HelloWorld")
        self.assertEqual(clean_text("Отлично!"), "Отлично!")
        self.assertEqual(clean_text("Может быть лучше."), "Может быть лучше.")
        self.assertEqual(clean_text("Не хватает деталей."), "Не хватает деталей.")
        self.assertEqual(clean_text(""), "")  # Пустая строка
        self.assertEqual(clean_text("Text with \x00 invalid \x01 characters"),
                         "Text with invalid characters")  # Удаление нескольких недопустимых символов
        self.assertEqual(clean_text("\x00\x01\x02"), "")  # Все недопустимые символы

    def test_process_data(self):
        # Нормальный случай
        input_data = [InputData(numeric_data=[1, 2, 3], text_data=["Good", "Nice"])]
        results = process_data(input_data)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ProcessingResult)
        self.assertEqual(results[0].average, 2.0)
        self.assertEqual(results[0].comments, "Good | Nice")

        # Тест с пустым InputData
        input_data_empty = []
        results_empty = process_data(input_data_empty)
        self.assertEqual(results_empty, [])  # Ожидаем пустой результат

        # Тест с InputData, содержащим пустые числовые и текстовые данные
        input_data_empty_fields = [InputData(numeric_data=[], text_data=[])]
        results_empty_fields = process_data(input_data_empty_fields)
        self.assertEqual(len(results_empty_fields), 1)
        self.assertIsInstance(results_empty_fields[0], ProcessingResult)
        self.assertEqual(results_empty_fields[0].comments, "")  # Нет комментариев

        # Тест с InputData, содержащим значения None
        input_data_with_none = [InputData(numeric_data=[None, 2, 3], text_data=["Comment"])]
        results_with_none = process_data(input_data_with_none)
        self.assertEqual(len(results_with_none), 1)
        self.assertIsInstance(results_with_none[0], ProcessingResult)
        self.assertEqual(results_with_none[0].average, 2.5)  # Среднее, игнорируя None
        self.assertEqual(results_with_none[0].comments, "Comment")  # Один комментарий

        # Тест с InputData, содержащим недопустимые типы
        input_data_invalid_type = [InputData(numeric_data=["string", 2, 3], text_data=["Comment"])]
        results_invalid_type = process_data(input_data_invalid_type)
        self.assertEqual(len(results_invalid_type), 1)
        self.assertIsInstance(results_invalid_type[0], ProcessingResult)

        # Тест с пустым текстовым полем
        input_data_empty_text = [InputData(numeric_data=[1, 2, 3], text_data=[""])]
        results_empty_text = process_data(input_data_empty_text)
        self.assertEqual(len(results_empty_text), 1)
        self.assertIsInstance(results_empty_text[0], ProcessingResult)
        self.assertEqual(results_empty_text[0].average, 2.0)
        self.assertEqual(results_empty_text[0].comments, "")  # Пустой комментарий

        # Тест с InputData, содержащим только None в числовых и текстовых полях
        input_data_only_none = [InputData(numeric_data=[None, None], text_data=[None])]
        results_only_none = process_data(input_data_only_none)
        self.assertEqual(len(results_only_none), 1)
        self.assertIsInstance(results_only_none[0], ProcessingResult)
        self.assertEqual(results_only_none[0].average, 0.0)  # Среднее должно быть 0.0
        self.assertEqual(results_only_none[0].comments, "")  # Нет комментариев

    def test_process_data_with_empty_input(self):
        input_data_empty = []
        results_empty = process_data(input_data_empty)
        self.assertEqual(results_empty, [])  # Ожидаем пустой результат

    def test_process_data_with_invalid_numeric_data(self):
        input_data_invalid = [InputData(numeric_data=["string", 2, 3], text_data=["Comment"])]
        results_invalid = process_data(input_data_invalid)
        self.assertEqual(len(results_invalid), 1)
        self.assertIsInstance(results_invalid[0], ProcessingResult)
        self.assertIsNotNone(results_invalid[0].error)  # Ошибка должна быть записана


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
