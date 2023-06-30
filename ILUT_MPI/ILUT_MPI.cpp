#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include "mpi.h"
#include <fstream>


template<class T>
class SparseMatrix
{
public:
    std::vector<T> values;               // vector of non-zero values
    std::vector<int> row_indices;        // indexes of the beginning of rows in the vector of values
    std::vector<int> column_indices;     // column indexes for each value in the value vector
    int rows_count, columns_count;       // number of rows and columns


    SparseMatrix() : rows_count(0), columns_count(0) {}
    ~SparseMatrix() = default;
    SparseMatrix(int rows, int columns)
        : values(),
        row_indices(rows + 1),
        column_indices(),
        rows_count(rows),
        columns_count(columns) {}


    double& operator()(int row, int column)
    {
        int n1 = row_indices[row + 1];

        for (int k = row_indices[row]; k < n1; k++)
            if (column_indices[k] == column) return values[k];

        // adding a new element to a vector
        int k = row_indices[row + 1];
        values.insert(values.begin() + k, 0);
        column_indices.insert(column_indices.begin() + k, column);

        for (int i = row + 1, n2 = row_indices.size(); i < n2; i++)
            row_indices[i]++;
        return values[k];
    }

    double operator()(int row, int column) const
    {
        int n = row_indices[row + 1];
        for (int k = row_indices[row]; k < n; ++k)
            if (column_indices[k] == column)
                return values[k];
        return 0;
    }
};


void printCSR(std::vector<int> row_ptr, std::vector<int> col_ind, std::vector<double> values, int n_rows, int n_columns, int num) {
    int k = 0;
    std::cout << "Print from process " << num << std::endl;
    std::cout << "------ "  << std::endl;
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_columns; j++) {
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
    std::cout << "------ " << std::endl;
}



SparseMatrix<double>ILUT(SparseMatrix<double> matrix, int n, double tau)
{
    //Initialize MPI
    MPI_Init(NULL, NULL);

    // Get total number of tasks
    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    // Get the task ID
    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    double start_time, end_time;
    start_time = MPI_Wtime();

    // initial matrix
    SparseMatrix<double> mat(n, n);

    int number_of_proc = 4;

    // matrix block size for each process
    int block_size = n / number_of_proc;

    // this is a block for each process
    SparseMatrix<double> chunk_matrix;

    // this is the processed block i get from other processes
    SparseMatrix<double> res;

    std::vector<int> size;
    std::vector<int> rowPtr_tmp;
    std::vector<int> colInd_tmp;
    std::vector<double> values_tmp;

    // create a matrix and send it to all processes
    if (task_id == 0)
    {
        // fill in the matrix
        mat.values = matrix.values;
        mat.row_indices = matrix.row_indices;
        mat.column_indices = matrix.column_indices;
        mat.rows_count = matrix.rows_count;
        mat.columns_count = matrix.columns_count;

        // print the original matrix
        // std::cout << "Start matrix from process 0" << std::endl;
        // printCSR(mat.row_indices, mat.column_indices, mat.values, n, n, 0);

        // sending the matrix to all processes
        for (int i = 1; i < number_of_proc; i++)
        {
            rowPtr_tmp.clear();
            values_tmp.clear();
            colInd_tmp.clear();

            // define block boundaries for i-th process
            const int start_row = i * block_size;
            int end_row;
            if (i == number_of_proc - 1)
            {
                end_row = n;
            }
            else {
                end_row = (i + 1) * block_size;
            }

            // copying array of strings for i-th process
            for (int j = start_row; j < end_row + 1; j++)
            {
                rowPtr_tmp.push_back(mat.row_indices[j] - mat.row_indices[start_row]);
            }

            // the bounds for the column numbers and values ​​for the i-th block
            auto start = mat.row_indices[start_row];
            auto end = mat.row_indices[end_row];

            for (int j = start; j < end; j++)
            {
                values_tmp.push_back(mat.values[j]);
                colInd_tmp.push_back(mat.column_indices[j]);
            }

            // compose the dimensions of each matrix-block array
            int s1 = end_row - start_row + 1;
            int s2 = end - start;
            int s3 = end - start;

            size = { s1, s2, s3 };
            MPI_Send(size.data(), 3, MPI_INT, i, 0, MPI_COMM_WORLD);

            MPI_Send(rowPtr_tmp.data(), s1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(colInd_tmp.data(), s2, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(values_tmp.data(), s3, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
        }

        // filling in the submatrix-block for process 0
        rowPtr_tmp.clear();
        values_tmp.clear();
        colInd_tmp.clear();

        for (int j = 0; j < block_size + 1; j++)
        {
            rowPtr_tmp.push_back(mat.row_indices[j]);
        }

        // the bounds for the column numbers and values ​​for the 0th block
        auto start = mat.row_indices[0];
        auto end = mat.row_indices[block_size];

        for (int j = start; j < end; j++)
        {
            values_tmp.push_back(mat.values[j]);
            colInd_tmp.push_back(mat.column_indices[j]);
        }

        chunk_matrix.column_indices = colInd_tmp;
        chunk_matrix.row_indices = rowPtr_tmp;
        chunk_matrix.values = values_tmp;
        chunk_matrix.rows_count = block_size;
        chunk_matrix.columns_count = n;

        // printing block from process 0
        // printCSR(chunk_matrix.row_indices, chunk_matrix.column_indices, chunk_matrix.values, chunk_matrix.rows_count, n, task_id);

    }

    if (task_id < number_of_proc && task_id != 0)
    {

        // the sizes of arrays of the matrix-block are stored here
        size.resize(3);
        MPI_Recv(size.data(), 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        rowPtr_tmp.resize(size[0]);
        MPI_Recv(rowPtr_tmp.data(), size[0], MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        colInd_tmp.resize(size[1]);
        MPI_Recv(colInd_tmp.data(), size[1], MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        values_tmp.resize(size[2]);
        MPI_Recv(values_tmp.data(), size[2], MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // this is a delay for printing to work properly
        for (int i = 0; i < task_id * 10000000; i++) {}

        chunk_matrix.column_indices = colInd_tmp;
        chunk_matrix.row_indices = rowPtr_tmp;
        chunk_matrix.values = values_tmp;
        chunk_matrix.rows_count = size[0] - 1;
        chunk_matrix.columns_count = n;

        // printing blocks from each process
        // printCSR(chunk_matrix.row_indices, chunk_matrix.column_indices, chunk_matrix.values, chunk_matrix.rows_count, n, task_id);
    }


    //---------------main algorithm---------------//

    // this is the string I am working with in the algorithm
    std::vector<double> row(n, 0);

    // this is the result matrix that i got from the last process
    SparseMatrix<double> result(n, n);

    // counter, how many processed lines the process has in the buffer (result)
    int count = 0;

    for (int i = 0; i < n; i++)
    {
        // determine whether this process should do calculations
        if (i / block_size == task_id || i / block_size > number_of_proc - 1 && task_id == number_of_proc - 1)
        {
            //determine which string it is in my local chunk
            int local_row = i % block_size;

            double norm_row_l = 0;

            for (int k = chunk_matrix.row_indices[local_row]; k < chunk_matrix.row_indices[local_row + 1]; k++)
            {
                row[chunk_matrix.column_indices[k]] = chunk_matrix.values[k];

                if (chunk_matrix.column_indices[k] < i)
                    norm_row_l += pow(chunk_matrix.values[k], 2);
            }
            norm_row_l = std::sqrt(norm_row_l);

            for (int k = 0; k < i; k++)
            {
                //need to understand if the process already has this line counted in its buffer
                //if there is, then continue counting
                //if not, you need to get it and only then continue to count
                if (k >= count)
                {
                    std::vector<double> r(n);

                    // need to get the processed string
                    MPI_Recv(r.data(), n, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // now it needs to be added to my array of processed strings
                    for (int j = 0; j < n; j++)
                    {
                        result(count, j) = r[j];
                    }
                    count++;
                }

                if (result(k, k) != 0 && row[k] != 0)
                {
                    if (std::abs(row[k]) < tau * norm_row_l)
                    {
                        row[k] = 0;
                    }
                    else {
                        const double rowk = (row[k] / result(k, k));

                        for (int j = result.row_indices[k] + 1; j < result.row_indices[k + 1]; j++)
                        {
                            if (result.column_indices[j] >= k + 1)
                            {
                                row[result.column_indices[j]] -= rowk * result.values[j];
                            }
                        }
                    }
                }
            }

            //finished reading the line
            // now we need to send the result further

            std::vector<MPI_Request>rec(number_of_proc - task_id);

            for (int i = task_id + 1; i < number_of_proc; i++)
            {
                MPI_Isend(row.data(), n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &rec[i - task_id - 1]);
                MPI_Wait(&rec[i - task_id - 1], MPI_STATUS_IGNORE);
            }

            // after the string has been sent to everyone and everyone has received it, you need to add it to your results chunk
            for (int j = 0; j < n; j++)
            {
                result(i, j) = row[j];
            }
            count++;
            row.clear();
            row.resize(n, 0);
        }
    }

    // turn off the light if you are the last
    if (task_id == number_of_proc - 1)
    {
        end_time = MPI_Wtime();
        std::cout << "Program execution time: " << (end_time - start_time) << " seconds" << std::endl;

        printCSR(result.row_indices, result.column_indices, result.values, result.rows_count, n, task_id);
    }


    MPI_Finalize();
    return result;
}


int main(int argc, char* argv[])
{
    int n = 8;
    double tau = 0.00001;

    SparseMatrix<double> mat(n, n);
    
    std::vector<int> rowPtr;
    std::vector<int> colInd;
    std::vector<double> values;

    mat.values = { 2, 2, 2, 4, 4, 1, 1, 4, 1, 1, 4, 1, 2, 3, 5, 4, 4, 1, 1, 4, 3, 4, 5, 5, 1, 3 };
    mat.row_indices = { 0,  1,  4,  7,  9, 13, 18, 21, 26 };
    mat.column_indices = { 4, 0, 1, 5, 0, 6, 7, 3, 7, 0, 3, 4, 7, 0, 1, 3, 4, 7, 0, 1, 3, 0, 3, 4, 6, 7 };
    mat.rows_count = n;
    mat.columns_count = n;

    ILUT(mat, n, tau);
    
    return 0;
}
