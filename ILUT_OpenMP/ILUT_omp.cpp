#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <omp.h>

// The structure of a sparse vector in CSR format
template<class T>
class SparseVector {
private:
    size_t size;
public:
    std::vector<T> values;       // values of non-zero elements
    std::vector<size_t> indices; // indexes of non-zero elements

    SparseVector() :size(0) {}
    ~SparseVector() = default;

    // Constructor with an initial size assignment
    SparseVector(size_t size) : values(0), indices(0), size(size) {}

    // Method for getting the vector size
    size_t get_size() const { return indices.size(); }

    // The [] operator for accessing the elements of a vector with a check for out-of-bounds
    double& operator[](size_t index)
    {
        if (index >= size) { throw std::exception("Index out of range"); }

        auto ind = std::find(indices.begin(), indices.end(), index);
        if (ind != indices.end())
        {
            return values[ind - indices.begin()];
        }
        else
        {
            indices.push_back(index);
            values.push_back(0);
            return values[values.size() - 1];
        }
    }

    // The [] operator for accessing elements of a vector with a check for out-of-bounds (constant version)
    double operator[](size_t index) const
    {
        if (index >= size) { throw std::exception("Index out of range"); }

        auto ind = std::find(indices.begin(), indices.end(), index);
        if (ind == indices.end())
        {
            return 0;
        }
        else
        {
            return values[ind - indices.begin()];
        }
    }

    // Overloading the equals operator to assign values
    SparseVector<T>& operator=(const SparseVector<T>& other)
    {
        values = other.values;
        indices = other.indices;
        return *this;
    }

    void zero_reset()
    {
        values.clear();
        indices.clear();
    }

};


template<class T>
class SparseMatrix
{
public:
    std::vector<T> values;                  // vector of non-zero values
    std::vector<size_t> row_indices;        // indexes of the beginning of rows in the vector of values
    std::vector<size_t> column_indices;     // column indexes for each value in the value vector
    size_t rows_count, columns_count;       // number of rows and columns


    SparseMatrix() : rows_count(0), columns_count(0) {}
    ~SparseMatrix() = default;
    SparseMatrix(size_t rows, size_t columns)
        : values(),
        row_indices(rows + 1),
        column_indices(),
        rows_count(rows),
        columns_count(columns) {}


    double& operator()(size_t row, size_t column)
    {
        size_t n1 = row_indices[row + 1];


        for (size_t k = row_indices[row]; k < n1; k++)
            if (column_indices[k] == column) return values[k];

        // adding a new element to a vector
        size_t k = row_indices[row + 1];
        values.insert(values.begin() + k, 0);
        column_indices.insert(column_indices.begin() + k, column);

        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = row + 1, n2 = row_indices.size(); i < n2; i++)
            row_indices[i]++;
        return values[k];
    }

    double operator()(size_t row, size_t column) const
    {
        size_t n = row_indices[row + 1];

        for (size_t k = row_indices[row]; k < n; ++k)
            if (column_indices[k] == column)
                return values[k];
        return 0;
    }
};

template<class T>
int ILU_tau(int n, double tau, SparseMatrix<T>& mat, SparseMatrix<T>& U, SparseMatrix<T>& L)
{
    // creating a matrix row
    SparseVector<T> row(n);

    for (int i = 0; i < n; i++)
    {
        // norm of the L part of the row
        double norm_row_l = 0;

        for (int k = mat.row_indices[i]; k < mat.row_indices[i + 1]; k++)
        {
            row[mat.column_indices[k]] = mat.values[k];

            // take the norm only of the L part to be consistent with the dropping rule applied in (10.)
            if (mat.column_indices[k] < i)
                norm_row_l += mat.values[k] * mat.values[k];
        }
        norm_row_l = std::sqrt(norm_row_l);


        for (int k = 0; k < i; k++)
        {
            if (U(k, k) != 0 && row[k] != 0)
            {
                if (std::abs(row[k]) < tau * norm_row_l)
                {
                    row[k] = 0;
                }
                else {
                    const T rowk = (row[k] / U(k, k));

                    for (int j = U.row_indices[k] + 1; j < U.row_indices[k + 1]; j++)
                    {
                        row[U.column_indices[j]] -= rowk * U.values[j];
                    }
                }
            }
        }

        // Copy values to L

        for (int j = 0; j < i; j++)
        {
            L(i, j) = row[j];
        }
        L(i, i) = 1;


        for (int j = i; j < n; j++)
        {
            U(i, j) = row[j];
        }
        U(i, i) = row[i];

        row.zero_reset();
    }

    return 0;
}


template<class T>
void printCSR(std::vector<size_t> row_ptr, std::vector<size_t> col_ind, std::vector<T> values, int n_rows) {
    int k = 0;
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_rows; j++) {
            if (k < col_ind.size() && col_ind[k] == j && k < row_ptr[i + 1]) {
                std::cout << values[k] << " ";
                k++;
            }
            else {
                std::cout << "0 ";
            }
        }
        std::cout << std::endl;
    }
}


template<typename T>
int read_array(std::vector<T>& v, std::string str)
{
    int size;

    std::ifstream file(str);

    if (!file.is_open())
    {
        std::cout << "Error opening file" << std::endl;
        return 1;
    }

    file >> size;
    v.resize(size);

    for (int i = 0; i < size; i++) { file >> v[i]; }

    file.close();

    return 0;
}


int main()
{
    int n = 20;
    double tau = 0.00001;

    std::vector<size_t> rowPtr;
    std::vector<double> values;
    std::vector<size_t> colInd;

    read_array<size_t>(rowPtr, "indptr20.txt");
    read_array<double>(values, "data20.txt");
    read_array<size_t>(colInd, "indices20.txt");

    SparseMatrix<double> mat(n, n);
    SparseMatrix<double> U(n, n);
    SparseMatrix<double> L(n, n);


    mat.values = values;
    mat.row_indices = rowPtr;
    mat.column_indices = colInd;
    mat.rows_count = n;
    mat.columns_count = n;

    auto start = std::chrono::steady_clock::now();

    ILU_tau(n, tau, mat, U, L);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;


    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "-------" << std::endl;
    printCSR(L.row_indices, L.column_indices, L.values, n);
    std::cout << "-------" << std::endl;
    printCSR(U.row_indices, U.column_indices, U.values, n);
    std::cout << "-------" << std::endl;


    return 0;
}
