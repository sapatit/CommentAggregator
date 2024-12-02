import argparse
import logging
import yaml
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


@dataclass
class InputData:
    numeric_data: List[float] = field(default_factory=list)  # Инициализация как пустой список
    text_data: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    average: Optional[float] = None
    comments: Optional[str] = None
    error: Optional[str] = None

    def __str__(self):
        return f"Average: {self.average}\nComments: {self.comments}\nError: {self.error}"


def process_data(input_data_list: List[InputData], save_to: Optional[Path] = None) -> List[ProcessingResult]:
    results = []
    for input_data in input_data_list:
        try:
            average = compute_average(input_data.numeric_data)
            comments = combine_comments(input_data.text_data)
            result = ProcessingResult(average=average, comments=comments)
        except Exception as e:
            result = ProcessingResult(error=str(e))

        results.append(result)

        # Сохранение результата, если указан путь
        if save_to:
            save_result(result, save_to)

    return results


def load_config(config_file: Path) -> Dict[str, InputData]:
    with config_file.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Loaded config is not a valid dictionary")

    input_data = {}
    for dataset_name, dataset_config in config.get('datasets', {}).items():
        numeric_data = [float(num) for num in dataset_config.get('numeric_data', [])]
        text_data = dataset_config.get('text_data', [])  # Получаем text_data, если он есть
        input_data[dataset_name] = InputData(numeric_data=numeric_data, text_data=text_data)

    return input_data


def compute_average(numeric_data: List[float]) -> float:
    if not isinstance(numeric_data, list):
        raise TypeError("numeric_data must be a list of numbers")

    cleaned_data = []
    for num in numeric_data:
        if num is None:
            continue  # Игнорируем None
        if not isinstance(num, (int, float)):
            raise TypeError("All elements in numeric_data must be numbers")
        cleaned_data.append(num)

    return sum(cleaned_data) / len(cleaned_data) if cleaned_data else 0.0


def combine_comments(text_data: List[str]) -> str:
    cleaned_comments = [clean_text(str(comment)) for comment in text_data if comment is not None]
    return " | ".join(cleaned_comments)


def save_result(result: ProcessingResult, file_path: Path) -> None:
    with file_path.open('w', encoding='UTF-8') as f:
        f.write(str(result))


def visualize_results(results: List[ProcessingResult], bins: int = 10) -> None:
    averages = [result.average for result in results if result.average is not None]
    plt.figure(figsize=(8, 6))
    plt.hist(averages, bins=bins, color='blue', alpha=0.7)
    plt.title("Distribution of Averages")
    plt.xlabel("Average")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def compare_results(results: Dict[str, List[ProcessingResult]], bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset_name, dataset_results in results.items():
        averages = [result.average for result in dataset_results if result.average is not None]
        ax.hist(averages, bins=bins, alpha=0.5, label=dataset_name)

    ax.set_title("Comparison of Averages")
    ax.set_xlabel("Average")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def clean_text(text):
    if text is None:
        return ""
    cleaned_text = re.sub(r'[^\w\s,.!?;:—-]', '', text)
    return ' '.join(cleaned_text.split())


def main(config_path: Path, save_path: Optional[Path]) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Загрузка конфигурации
    input_data_dict = load_config(config_path)

    # Обработка данных
    results = {}
    for dataset_name, input_data in input_data_dict.items():
        result = process_data([input_data], save_to=save_path)
        results[dataset_name] = result

    # Визуализация результатов
    for dataset_name, dataset_results in results.items():
        logging.info(f"Результаты для набора данных '{dataset_name}':")
        for res in dataset_results:
            print(res)

    # Визуализация распределения средних значений
    visualize_results([res for dataset_results in results.values() for res in dataset_results])

    # Сравнение результатов
    compare_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка данных из конфигурационного файла.")
    parser.add_argument('config', type=Path, help='Путь к конфигурационному файлу (YAML)')
    parser.add_argument('--save', type=Path, help='Путь к файлу для сохранения результатов')

    args = parser.parse_args()
    main(args.config, args.save)
