#include <iostream>
#include <omp.h>
#include <numa.h>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>       /* time */
#include <math.h>   
#include <vector> 
using namespace std; 

//this function mutiply L and U to get LU
void computeLU(double **upper, double **lower, double **lu, int n){
    //matrix multiplication algorithm
    #pragma omp for
    for (size_t i = 0; i < n; i++) {
        //values of lu are computed row by row
        double *lu_row=(double *)numa_alloc_local(n*sizeof(double));
        for (size_t j = 0; j < n; j++) {
            lu_row[j]=0.0;
            for (size_t k = 0; k < n; k++){
                lu_row[j] += lower[i][k] * upper[k][j];
            }
        }
        lu[i]=lu_row;
    }
}

void computePA(double **a, int *pi, double **pa, int n){
    #pragma omp for
    for (size_t i=0; i < n; i++){
        int k=pi[i];
        pa[i]=a[k];
    }
}

//This function check whether the LU decomposition is accurate by computing the sum of Euclidean norms of the columns 
//of the residual matrix (this sum is known as the L2,1 norm computed as PA - LU 
//It prints out value of the L2,1 norm of the residual, which is expected to be very small
double checkResult(double **a, double **upper, double **lower, int *pi, int n){
    
    //instantiate sum, and matrices pa and lu
    double sum;
    double *pa[n];
    double *lu[n];
    
    //invoke paralle region
    //multiplication of L&U and P&A will be done in parallel
    #pragma omp parallel shared(a, upper, lower, pi, n, pa, lu, sum) default(none)
    {   
        //mutiply L and U to get LU
        computeLU(upper, lower, lu, n);
        
        //mutiply P and A to get PA
        computePA(a, pi, pa, n);
        
        //each iteration compute Euclidean norm of one column
        #pragma omp for reduction(+: sum)
        for (size_t j=0; j < n; j++){
            double colsum;
            for (size_t i=0; i<n; i++){
                colsum+=pow(pa[i][j]-lu[i][j], 2);
            }
            
            //reduction that stores all colsum
            sum+=sqrt(colsum);
        }
        
    }      

    printf("residual sum %f \n", sum);
    return sum;
}

//task of each iteration in lu decomposition that is done in serial
void serialPart(double **a, double **upper, double **lower, int *pi, int n, size_t k){
    size_t i, j;
    double max=0;
    size_t kk;
    for(i=k; i<n; i++){

        if(max<fabs(a[i][k])){
            max=fabs(a[i][k]);
            kk=i;
        }
    }
    if(max==0){
        throw "ERROR: singular matrix";
    }

    int temppi=pi[k];
    pi[k]=pi[kk];
    pi[kk]=temppi;

    double *temp_a=a[k];
    a[k]=a[kk];
    a[kk]=temp_a;

    for(i=0; i<k; i++){
        double templ=lower[k][i];
        lower[k][i]=lower[kk][i];
        lower[kk][i]=templ;
    }

    upper[k][k]=a[k][k];
}

//task of each iteration in lu decomposition that is done in parallel
//two parallel for is used because the operation in the second for loop is dependent on values computed in the first for loop
void parallelpart(double **a, double **upper, double **lower, int n, int nthreads, size_t k){
    #pragma omp parallel for shared(a, upper, lower, n, k) num_threads(nthreads) default(none) proc_bind(spread)
    for(size_t i=k+1; i<n; i++){
        lower[i][k]=a[i][k]/upper[k][k];
        upper[k][i]=a[k][i];
    }

    #pragma omp parallel for shared(a, upper, lower, n, k) num_threads(nthreads) default(none) proc_bind(spread) 
    for(size_t i=k+1; i<n; i++){
        for(size_t j=k+1; j<n; j++){
            a[i][j]-=(lower[i][k]*upper[k][j]);
        }
    }
    
}

//this function take populated random matrix a, initialized matricies upper, lower, initilized vector pi, matrix size n and number of threads nthreads as input
void decompose(double **a, double **upper, double **lower, int *pi, int n, int nthreads){    
    //iterate n times
    for(size_t k=0; k<n; k++){
        //tasks to be done in serial
        serialPart(a, upper, lower, pi, n, k);
        //tasks to be done in parallel
        parallelpart(a, upper, lower, n, nthreads, k);
    }
}


//main functino
int main(int argc, const char* argv[])
{   
    //determine matrix size and number of threads from input
    int n=atoi(argv[1]);
    int nthreads = atoi(argv[2]);

    //initialize matricies
    double *a[n];
    double *acopy[n];
    double *upper[n];
    double *lower[n];
    int pi[n];

    //initialize drand48 and set seed
    struct drand48_data state;
    srand48_r(omp_get_wtime()*114514, &state);

    //populate matricies in parallel
    //each threads initialize a number of rows and allocate data to the memory of the socket it is running on
    #pragma omp parallel for shared(a, upper, lower, n, acopy, pi, state, nthreads) num_threads(nthreads) default(none) proc_bind(spread) schedule(static, 1)
    for(int i=0; i<n; i++){
        //populate pi
        pi[i]=i;
        //initialize row
        double *u_row = (double *)numa_alloc_local(n*sizeof(double));
        double *l_row = (double *)numa_alloc_local(n*sizeof(double));
        double *a_row = (double *)numa_alloc_local(n*sizeof(double));
        double *acopy_row = (double *)numa_alloc_local(n*sizeof(double));

        //populate each column
        for(int j=0; j<n; j++){
            //assign random numbers to matrix A
            drand48_r(&state, &a_row[j]);
            acopy_row[j]=a_row[j];
            //assign 1 to diagonal of lower matrix
            if(j==i){
                l_row[j]=1;
            }
        }

        //assign row to current index
        a[i]=a_row;
        acopy[i]=acopy_row;
        upper[i]=u_row;
        lower[i]=l_row;
    }

    //decompose matrix a and record the time
    double start = omp_get_wtime();
    decompose(a, upper, lower, pi, n, nthreads);
    double end = omp_get_wtime();
    printf("%d %lld %f\n", nthreads, n, end-start);
    
    //check results of decomposition
    checkResult(acopy, upper, lower, pi, n);

    return 0;
}
