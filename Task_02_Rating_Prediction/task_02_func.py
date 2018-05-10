"""
File of functions to be used in Jupyter Notebook "task_02_rating_prediction"
Group 16, Mining Massive Datasets[IN2323], Summer 2018
"""

def remove_rows(matrix, min_entries):
    """
    Removes the rows from the input matrix which have atmost min_entries nonzero entries.
    
    Parameters
    ----------
    matrix      : sp.spmatrix, shape [N, D]
                  The input matrix from which rows should be removed.
    min_entries : int
                  Minimum number of nonzero elements per row.
    
    Returns
    -------
    matrix      : sp.spmatrix, shape [N', D]
                  The updated matrix, where every row now has more than min_entries 
                  nonzero entries, such that N' <= N            
    """
    
    change = False
    rows_to_keep = []
    
    #iterate over every row of the matrix
    for row in range(matrix.shape[0]):
        #if this row has more than min_entries nonzero entries 
        #it is added to a list of rows which are kept
        if matrix[row].count_nonzero() > min_entries:
            rows_to_keep.append(row)
    
    #determine wether there were rows removed or not
    #which is important for having a termination condition of the recursion
    #in the preproc_matrix function
    if len(rows_to_keep) < matrix.shape[0]:
        change = True
    
    #update the matrix by only keeping the rows from the list
    matrix = matrix[rows_to_keep]
    
    return matrix, change

def preproc_matrix(matrix, min_entries):
    
    """
    Preprocesses the input matrix by recursivly removing 
    rows and columns from the input matrix which have atmost min_entries nonzero entries.
    
    Parameters
    ----------
    matrix      : sp.spmatrix, shape [N, D]
                  The input matrix to be preprocessed.
    min_entries : int
                  Minimum number of nonzero elements per row and columns
                  
    Returns
    -------
    matrix      : sp.spmatrix, shape [N', D']
                  The pre-processed matrix, where N' <= N and D' <= D              
    """
    
    #first remove all rows which have atmost min_entries nonzero entries
    matrix = remove_rows(matrix, min_entries)[0]
    #do the same for all columns (by applying the same function on the transposed matrix)
    #keep track if columns were removed for recursion
    matrix_T, change = remove_rows(matrix.transpose(), min_entries)
    matrix = matrix_T.transpose()
    
    #if columns were removed from the matrix, it is necessary to do another recursion
    if change:
        matrix, change = preproc_matrix(matrix, min_entries)
    
    return matrix, change