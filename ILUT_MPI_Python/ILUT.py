
import scipy.sparse as sp

import numpy as np
from SparseVector import SparseVector
from SparseMatrix import SparseMatrix
from add_sparse_vectors import add_sparse_vectors

from mpi4py import MPI
import numpy as np

def printCSR(row_ptr, col_ind, values, n_rows, n_col, proc):
    k = 0
    print("Print matix from ", proc)
    print("---------------")
    for i in range(n_rows):
        for j in range(n_col):
            if k < len(col_ind) and col_ind[k] == j and k < row_ptr[i + 1]:
                print(np.round(values[k], 2), end=" ")
                k += 1
            else:
                print(0, end=" ")
        print()
    print("---------------")


def ILUT(matrix, n, tau):

    # Initialize MPI environment
    comm = MPI.COMM_WORLD

    num_tasks = comm.Get_size()

    task_id = comm.Get_rank()

    start_time, end_time = 0, 0

    start_time = MPI.Wtime()

    # Initial matrix
    mat = SparseMatrix(n, n)

    number_of_proc = 2

    # matrix block size for each process
    block_size = n // number_of_proc

    # this is a block for each process
    chunk_matrix = SparseMatrix(n, n)

    # this is the processed block i get from other processes
    res = SparseMatrix(n, n)

    size = np.array(0)
    rowPtr_tmp = np.array(0)
    colInd_tmp = np.array(0)
    values_tmp = np.array(0)

    # create a matrix and send it to all processes
    if task_id == 0:
        # fill in the matrix
        mat.values = matrix.values.copy()
        mat.row_indices = matrix.row_indices.copy()
        mat.column_indices = matrix.column_indices.copy()
        mat.rows_count = matrix.rows_count
        mat.columns_count = matrix.columns_count

        # sending the matrix to all processes
        for i in range(1, number_of_proc):

            rowPtr_tmp = np.array([],dtype=int)
            values_tmp = np.array([], dtype=float)
            colInd_tmp = np.array([], dtype=int)

            # define block boundaries for i-th process
            start_row = i * block_size
            if i == number_of_proc - 1:
                end_row = n
            else:
                end_row = (i + 1) * block_size

            #print(start_row, end_row + 1)
            # copying array of strings for i-th process
            for j in range(start_row, end_row + 1):
                rowPtr_tmp = np.append(rowPtr_tmp, mat.row_indices[j] - mat.row_indices[start_row])


            # the bounds for the column numbers and values for the i-th block
            start = mat.row_indices[start_row]
            end = mat.row_indices[end_row]

            for j in range(start, end):
                values_tmp = np.append(values_tmp, mat.values[j])
                colInd_tmp = np.append(colInd_tmp, mat.column_indices[j])


            # compose the dimensions of each matrix-block array
            s1 = end_row - start_row + 1
            s2 = end - start
            s3 = end - start

            size = np.array([s1, s2, s3], dtype=int)
            comm.Send(size, dest=i, tag=0)

            rowPtr_tmp =np.array(rowPtr_tmp, dtype=int)
            colInd_tmp =colInd_tmp.astype(int)
            values_tmp =values_tmp.astype(float)

            comm.Send(rowPtr_tmp, dest=i, tag=1)
            comm.Send(colInd_tmp, dest=i, tag=2)
            comm.Send(values_tmp, dest=i, tag=3)


        # filling in the submatrix-block for process 0
        rowPtr_tmp = np.array([], dtype=int)
        values_tmp = np.array([], dtype=float)
        colInd_tmp = np.array([], dtype=int)

        for j in range(block_size + 1):
            rowPtr_tmp = np.append(rowPtr_tmp, mat.row_indices[j])
            

        # the bounds for the column numbers and values for the 0th block
        start = mat.row_indices[0]
        end = mat.row_indices[block_size]

        for j in range(start, end):
            values_tmp = np.append(values_tmp, mat.values[j])
            colInd_tmp = np.append(colInd_tmp, mat.column_indices[j])
            
        chunk_matrix.column_indices = colInd_tmp.copy()
        chunk_matrix.row_indices = rowPtr_tmp.copy()
        chunk_matrix.values = values_tmp.copy()
        chunk_matrix.rows_count = block_size
        chunk_matrix.columns_count = n

        #printCSR(rowPtr_tmp,colInd_tmp, values_tmp, block_size, n, task_id )

    if task_id < number_of_proc and task_id != 0:
       
        # the sizes of arrays of the matrix-block are stored here
        size = np.empty(3 + 1, dtype=int)
        comm.Recv(size, source=0, tag=0)

        rowPtr_tmp = np.empty(size[0] + 1 , dtype=int)
        comm.Recv(rowPtr_tmp, source=0,tag=1)

        colInd_tmp = np.empty(size[1] + 1, dtype=int)
        comm.Recv(colInd_tmp, source=0, tag=2)

        values_tmp = np.empty(size[2] + 1, dtype=float)
        comm.Recv(values_tmp, source=0, tag=3)
    
        # this is a delay for printing to work properly
        for i in range(task_id * 1000000): pass
        
        chunk_matrix.column_indices = colInd_tmp.copy()
        chunk_matrix.row_indices = rowPtr_tmp.copy()
        chunk_matrix.values = values_tmp.copy()
        chunk_matrix.rows_count = size[0] - 1
        chunk_matrix.columns_count = n
        #printCSR(rowPtr_tmp,colInd_tmp, values_tmp, size[0] - 1, n, task_id )

    #if(task_id < number_of_proc):
    #    printCSR(chunk_matrix.row_indices,chunk_matrix.column_indices, chunk_matrix.values, chunk_matrix.rows_count, chunk_matrix.columns_count, task_id )
        
        # ---------------main algorithm---------------#


    # this is the string I am working with in the algorithm
    row = np.zeros(n)

    # this is the result matrix that i got from the last process
    result = SparseMatrix(n, n)

    # counter, how many processed lines the process has in the buffer (result)
    count = 0

    
    for i in range(n):
        # determine whether this process should do calculations
        if i // block_size == task_id or (i // block_size > number_of_proc - 1 and num_tasks == number_of_proc - 1):

            # determine which string it is in my local chunk
            local_row = i % block_size

            norm_row_l = 0

            for k in range(chunk_matrix.row_indices[local_row], chunk_matrix.row_indices[local_row + 1]):
                row[chunk_matrix.column_indices[k]] = chunk_matrix.values[k]

                if chunk_matrix.column_indices[k] < i:
                    norm_row_l += pow(chunk_matrix.values[k], 2)
            norm_row_l = np.sqrt(norm_row_l)
 

            for k in range(i):
                # need to understand if the process already has this line counted in its buffer
                # if there is, then continue counting
                # if not, you need to get it and only then continue to count
                if k >= count:

                    r = np.empty(n + 1, dtype=float)
                    # need to get the processed string
                    comm.Recv(r, source=MPI.ANY_SOURCE, tag=0)
 
                    # now it needs to be added to my array of processed strings
                    for j in range(n):
                        result[count, j] = r[j]
                    count += 1

                if result[k, k] != 0 and row[k] != 0:

                    if np.abs(row[k]) < tau * norm_row_l:
                        row[k] = 0
                    else:
                        rowk = row[k] / result[k, k]

                        for j in range(result.row_indices[k] + 1, result.row_indices[k + 1]):
                            if result.column_indices[j] >= k + 1:
                                row[result.column_indices[j]] -= rowk * result.values[j]

            # finished counting the line
            # now we need to send the result further

            for j in range(task_id + 1, number_of_proc):
                req  = comm.Isend(row, dest=j, tag=0)
                req.Wait()

            # after the string has been sent to everyone and everyone has received it, you need to add it to your results chunk
            for j in range(n):
                result[i, j] = row[j]

            count += 1
            row.fill(0)

    # turn off the light if you are the last
    if task_id == number_of_proc - 1:
        end_time = MPI.Wtime()
        print("Program execution time: ", (end_time - start_time), " seconds")

        printCSR(result.row_indices, result.column_indices, result.values, result.rows_count, n, task_id)

    return result
    

n = 50

np.random.seed(42)
A = np.random.choice(a=[0, 1, 2, 3, 4, 5], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], size=(n, n))
A = sp.csr_matrix(A)

tau = 0.001

mat = SparseMatrix(n, n)


mat.values = list(A.data)
mat.row_indices = list(A.indptr)
mat.column_indices = list(A.indices)


mat.rows_count = n
mat.columns_count = n

ILUT(mat, n, tau)


