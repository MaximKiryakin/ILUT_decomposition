class SparseVector:

    # Конструктор с заданием начального размера
    def __init__(self, size=0):
        self.size = size
        self.values = []
        self.indices = []

    # Метод для получения размера вектора
    def get_size(self):
        return len(self.indices)

    # Оператор [] для доступа к элементам вектора с проверкой на выход за границы
    def __getitem__(self, index):
        if index >= self.size:
            raise Exception("Index out of range")

        ind = self.indices.index(index) if index in self.indices else -1

        if ind != -1:
            return self.values[ind]
        else:
            return 0

    def __setitem__(self, index: int, value: float):
        if index >= self.size:
            raise Exception("Index out of range")

        ind = self.indices.index(index) if index in self.indices else -1
        if ind == -1:
            self.indices.append(index)
            self.values.append(value)
        else:
            self.values[ind] = value

    # Перегрузка оператора равно для присваивания значений
    def __eq__(self, other):
        self.values = other.values
        self.indices = other.indices
        self.size = other.size
        return self