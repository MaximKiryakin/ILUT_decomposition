class SparseMatrix:
    def __init__(self, rows, columns):
        self.values = []
        self.row_indices = [0] * (rows + 1)
        self.column_indices = []
        self.rows_count = rows
        self.columns_count = columns

    def __getitem__(self, index):

        row, column = index

        n1 = self.row_indices[row + 1]

        for k in range(self.row_indices[row], n1):
            if self.column_indices[k] == column:
                return self.values[k]
        return 0

    def __setitem__(self, index, value):
        row, column = index
        n1 = self.row_indices[row + 1]

        for k in range(self.row_indices[row], n1):
            if self.column_indices[k] == column:
                self.values[k] = value
                return

        # добавление нового элемента в вектор
        k = self.row_indices[row + 1]
        self.values.insert(k, value)
        self.column_indices.insert(k, column)

        for i in range(row + 1, len(self.row_indices)):
            self.row_indices[i] += 1

    def __call__(self, row, column):
        return self.__getitem__((row, column))

    def __str__(self):
        output = ''
        for i in range(self.rows_count):
            for j in range(self.columns_count):
                output += str(self[i, j]) + ' '
            output += '\n'
        return output
