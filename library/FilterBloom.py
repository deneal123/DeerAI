from bitarray import bitarray
import mmh3

class BloomFilter:
    def __init__(self, size, hash_count):
        """
        Инициализация фильтра Блума.

        :param size: Размер битового массива.
        :param hash_count: Количество хэш-функций.
        """
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, item):
        """
        Добавление элемента в фильтр Блума.

        :param item: Элемент для добавления.
        """
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            self.bit_array[index] = 1

    def __contains__(self, item):
        """
        Проверка наличия элемента в фильтре Блума.

        :param item: Элемент для проверки.
        :return: True, если элемент возможно присутствует в фильтре Блума, иначе False.
        """
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            if self.bit_array[index] == 0:
                return False
        return True


# Пример использования:
data = ["apple", "banana", "orange", "grape", "watermelon"]
bloom_filter = BloomFilter(size=100, hash_count=5)

for item in data:
    bloom_filter.add(item)

query = "coco"
print(f"Данные '{query}' {'есть' if query in bloom_filter else 'отсутствуют'} в фильтре Блума.")