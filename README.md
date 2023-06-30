# Реализация алгоритма ILUT разложения

Данный алгоритм реализует ILU-разложение матрицы, представленной в разреженном формате `csr`, с применением порогового значения `tau`. Разложение проводится в параллельном режиме с помощью OpenMP и MPI.
MPI версия была реализована на языках `C++` и `Puthon`

Алгоритм был взят из книги Саада. Файл `ILUT.pdf` 
  