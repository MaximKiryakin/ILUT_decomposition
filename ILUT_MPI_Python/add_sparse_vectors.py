from SparseVector import SparseVector
# Функция для сложения двух разреженных векторов в CSR формате
def add_sparse_vectors(v1, v2, a, b, n):

    result = SparseVector(n)
    i, j = 0, 0

    while i < v1.get_size() and j < v2.get_size():
        if v1.indices[i] == v2.indices[j]:
            # Если индексы элементов совпадают, сложить значения
            sum = a * v1.values[v1.indices[i]] + b * v2.values[v2.indices[j]]
            if sum != 0:
                result[v1.indices[i]] = sum
            i += 1
            j += 1

        elif v1.indices[i] < v2.indices[j]:
            # Если индекс элемента в первом векторе меньше, записать его значение в результирующий вектор
            result[v1.indices[i]] = a * v1.values[i]
            i += 1
        else:
            # Если индекс элемента во втором векторе меньше, записать его значение в результирующий вектор
            result[v2.indices[j]] = b * v2.values[j]
            j += 1

    # Добавить оставшиеся элементы
    for i in range(i, v1.get_size()):
        result.values.append(a * v1.values[i])
        result.indices.append(v1.indices[i])
    for j in range(j, v2.get_size()):
        result.values.append(b * v2.values[j])
        result.indices.append(v2.indices[j])

    return result

