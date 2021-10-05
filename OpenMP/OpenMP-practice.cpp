#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <vector>

//populate matrix a and b
void generateMatrix(double *a, double *b, int n){
  srand(time(NULL)*114514);  
  for(int i=0; i<n*n; i++){
    a[i] = (rand() / (double)RAND_MAX);
    //a[i]=i;
    b[i] = (rand() / (double)RAND_MAX);
    //b[i]=i;
  }
}

//multiply a and b blocks and store the result at cblock
void serial_multiply(double *ablock, double *bblock, double *cblock, int block_size){
  for (int i = 0; i < block_size; i++) {
    for (int j = 0; j < block_size; j++) {
      for (int k = 0; k < block_size; k++){
        cblock[i*block_size+j] += ablock[i*block_size+k] * bblock[k*block_size+j];
      }
    }
  }
}

//compute the sum of Euclidean norms of each column in the residual matrix given by C’-C 
void compute_l2norm(double *c1, double *c2, int n){
  double sum;
  for (int j=0; j < n; j++){
    double colsum;
    for (int i=0; i<n; i++){
      colsum+=pow(c1[i*n+j]-c2[i*n+j], 2);
    }
    sum+=sqrt(colsum);
  }
  printf("residual sum %f \n", sum);
}

//restore matrix to regular arrangement
//this function is necessary because I assume my 2.5d matrix multiply work on matricies that are sorted block by block
//i need to restore matrix a and b to orignal arrangement so i can multiply them serially to compare results of my 2.5d algo
void restore_matrix(double *matrix1, double *matrix2, int n, int block_size, int num_blocks){
  int counter=0;
  int gap=n-block_size;
  int block_per_line=n/block_size;
  for (int block_no=0; block_no < num_blocks; block_no++){
    int start=n*block_size*(block_no/block_per_line)+block_size*(block_no%block_per_line);
    for(int i=0; i<block_size; i++){
      for(int j=0; j<block_size; j++){
        matrix2[start+j]=matrix1[counter];
        counter++;
      }
      start+=(gap+block_size);
    }
  }

}

//check the accuracy of my result c matrix
void check_result(double *a, double *b, double *c, int n, int block_size){
  double *acopy = (double *)malloc(n*n*sizeof(double));
  double *bcopy = (double *)malloc(n*n*sizeof(double));
  double *ccopy = (double *)malloc(n*n*sizeof(double));

  int num_blocks=pow(n/block_size, 2);
    
  //need to restore a and b to normal arrangment to make serial mutiplication
  restore_matrix(a, acopy, n, block_size, num_blocks);
  restore_matrix(b, bcopy, n, block_size, num_blocks);

  double *serial_c = (double *)malloc(n*n*sizeof(double));
  for(int i=0; i<n*n; i++){
    serial_c[i]=0;
  }
    
  serial_multiply(acopy, bcopy, serial_c, n);

  //restore c to normal arrangement and store the result at c_copy
  restore_matrix(c, ccopy, n, block_size, num_blocks);

/**
  for(int i=0; i<n*n; i++){
    printf("%f ", serial_c[i]);
    }
  printf("\n");
  for(int i=0; i<n*n; i++){
    printf("%f ",ccopy[i]);
  }
  printf("\n");
*/
  printf("serial_c %f \n", serial_c[1]);
  printf("c %f \n", ccopy[1]);
    
  compute_l2norm(serial_c, ccopy, n);
  free(serial_c);
  free(acopy);
  free(bcopy);
  free(ccopy);
}

//3d matrix multiply algorithm
//I built my 2.5d upon this function
//grader can disregard this funciton
void multiply_3d(double *a, double *b, double *c, int n, int pdim){

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int ndims=3;
  int dimension=pdim;
  int block_size=n/dimension;
  
  int dims[ndims];
  dims[0]=dims[1]=dims[2]=dimension;
  int periods[ndims];
  periods[0] = periods[1] = periods[2] = 1;

  MPI_Comm comm3d;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm3d);

  int rank3d, cart_rank;
  int coords3d[ndims];  
  MPI_Comm_rank(comm3d, &rank3d);
  MPI_Cart_coords(comm3d, world_rank, ndims, coords3d); 
  MPI_Cart_rank(comm3d, coords3d, &cart_rank);

  double* ablock=(double *)malloc(block_size*block_size*sizeof(double));
  double* bblock=(double *)malloc(block_size*block_size*sizeof(double));
  double* cblock=(double *)malloc(block_size*block_size*sizeof(double));

  int keep_dims_xy[3];
  keep_dims_xy[0]=keep_dims_xy[1]=1;
  keep_dims_xy[2]=0;
  MPI_Comm comm_xy;
  MPI_Cart_sub(comm3d, keep_dims_xy, &comm_xy);
  int rank_xy;
  MPI_Comm_rank(comm_xy, &rank_xy);

  int keep_dims_yz[3];
  keep_dims_yz[1]=keep_dims_yz[2]=1;
  keep_dims_yz[0]=0;
  MPI_Comm comm_yz;
  MPI_Cart_sub(comm3d, keep_dims_yz, &comm_yz);
  int rank_yz;
  MPI_Comm_rank(comm_yz, &rank_yz);

  int keep_dims_xz[3];
  keep_dims_xz[0]=keep_dims_xz[2]=1;
  keep_dims_xz[1]=0;
  MPI_Comm comm_xz;
  MPI_Cart_sub(comm3d, keep_dims_xz, &comm_xz);
  int rank_xz;
  MPI_Comm_rank(comm_xz, &rank_xz);

  int keep_dims_x[3];
  keep_dims_x[0]=1;
  keep_dims_x[1]=keep_dims_x[2]=0;
  MPI_Comm comm_x;
  MPI_Cart_sub(comm3d, keep_dims_x, &comm_x);
  int rank_x;
  MPI_Comm_rank(comm_x, &rank_x);

  int keep_dims_y[3];
  keep_dims_y[1]=1;
  keep_dims_y[0]=keep_dims_y[2]=0;
  MPI_Comm comm_y;
  MPI_Cart_sub(comm3d, keep_dims_y, &comm_y);
  int rank_y;
  MPI_Comm_rank(comm_y, &rank_y);

  int keep_dims_z[3];
  keep_dims_z[2]=1;
  keep_dims_z[1]=keep_dims_z[0]=0;
  MPI_Comm comm_z;
  MPI_Cart_sub(comm3d, keep_dims_z, &comm_z);
  int rank_z;
  MPI_Comm_rank(comm_z, &rank_z);

  //braodcast a
  if(rank_z==0){
    MPI_Scatter(a, block_size*block_size, MPI_DOUBLE, ablock, block_size*block_size, MPI_DOUBLE, 0, comm_xy);
  }
  MPI_Bcast(ablock, block_size*block_size, MPI_DOUBLE, 0, comm_z);

  //braodcast b
  if(rank_x==0){
    MPI_Scatter(b, block_size*block_size, MPI_DOUBLE, bblock, block_size*block_size, MPI_DOUBLE, 0, comm_yz);
  }

  MPI_Bcast(bblock, block_size*block_size, MPI_DOUBLE, 0, comm_x);
 
  //serial multiply at each block
  serial_multiply(ablock, bblock, cblock, block_size);

  //reduce c
  double *reduced_cblock=(double *)malloc(block_size*block_size*sizeof(double));
  MPI_Reduce(cblock, reduced_cblock, block_size*block_size, MPI_DOUBLE, MPI_SUM, 0, comm_y);

  if(rank_y==0){
    MPI_Gather(reduced_cblock, block_size*block_size, MPI_DOUBLE, c, block_size*block_size, MPI_DOUBLE, 0, comm_xz);
  }

  free(ablock);
  free(bblock);
  free(cblock);
  free(reduced_cblock);
}

//multiply two matrix with 2.5d algorithm and store the result at c
void multiply_25d(double *a, double *b, double *c, int n, int depth){

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  //initializing 3d communicator
  int ndims=3;
  int dimension=std::sqrt(world_size/depth);
  int block_size=n/dimension;
  int block_squre=block_size*block_size;
  
  int dims[ndims];
  dims[0] = dims[1] = dimension;
  dims[2] = depth;
  int periods[ndims];
  periods[0] = periods[1] = periods[2] = 1;

  MPI_Comm comm3d;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm3d);

  int rank3d, cart_rank;
  int coords3d[ndims];  
  MPI_Comm_rank(comm3d, &rank3d);
  MPI_Cart_coords(comm3d, world_rank, ndims, coords3d); 
  MPI_Cart_rank(comm3d, coords3d, &cart_rank);

  //allocate memory to store blocks of matrix 
  double* ablock=(double *)malloc(block_squre*sizeof(double));
  double* bblock=(double *)malloc(block_squre*sizeof(double));
  double* cblock=(double *)malloc(block_squre*sizeof(double));
  double *reduced_cblock=(double *)malloc(block_squre*sizeof(double));

  //initiate xy plane communicator (for processors with same z and different xy)
  int keep_dims_xy[3];
  keep_dims_xy[0]=keep_dims_xy[1]=1;
  keep_dims_xy[2]=0;
  MPI_Comm comm_xy;
  MPI_Cart_sub(comm3d, keep_dims_xy, &comm_xy);
  int rank_xy;
  MPI_Comm_rank(comm_xy, &rank_xy);

  //initiate z dimension communicator (for processors with same xy and different z)
  int keep_dims_z[3];
  keep_dims_z[2]=1;
  keep_dims_z[1]=keep_dims_z[0]=0;
  MPI_Comm comm_z;
  MPI_Cart_sub(comm3d, keep_dims_z, &comm_z);
  int rank_z;
  MPI_Comm_rank(comm_z, &rank_z);

  //P000 scatter a and b block by block to Pxy0
  //Since i assume my blocks are sorted block by block, i can just scatter matrix directly with size=num of elements in a block
  if(rank_z==0){
    MPI_Scatter(a, block_squre, MPI_DOUBLE, ablock, block_squre, MPI_DOUBLE, 0, comm_xy);
    MPI_Scatter(b, block_squre, MPI_DOUBLE, bblock, block_squre, MPI_DOUBLE, 0, comm_xy);
  }

  //Pxy0 broadcasts Axy and Bxy to all Pxyz 
  MPI_Bcast(ablock, block_squre, MPI_DOUBLE, 0, comm_z);
  MPI_Bcast(bblock, block_squre, MPI_DOUBLE, 0, comm_z);

  //compute round of shift
  int round_shift=sqrt(world_size / pow(depth, 3));

  //compute sending and reciving coords for initial circular shift
  int r=(coords3d[1]+coords3d[0]-coords3d[2]*round_shift)%dimension;
  int s=(coords3d[1]-coords3d[0]+coords3d[2]*round_shift)%dimension;
  int s1=(coords3d[0]-coords3d[1]+coords3d[2]*round_shift)%dimension;

  //get ranks according to coordinates
  int asend, arecv, bsend, brecv;
  int asendcoord[ndims]={coords3d[0], s, coords3d[2]};
  MPI_Cart_rank(comm3d, asendcoord, &asend);
  int bsendcoord[ndims]={s1, coords3d[1], coords3d[2]};
  MPI_Cart_rank(comm3d, bsendcoord, &bsend);
  
  int arecvcoord[ndims]={coords3d[0], r, coords3d[2]};
  MPI_Cart_rank(comm3d, arecvcoord, &arecv);
  int brecvcoord[ndims]={r, coords3d[1], coords3d[2]};
  MPI_Cart_rank(comm3d, brecvcoord, &brecv);

  MPI_Status status[2];

  //shift blocks
  MPI_Sendrecv_replace(ablock, block_squre, MPI_DOUBLE, asend, 1, arecv, 1, comm3d, &status[0]);
  MPI_Sendrecv_replace(bblock, block_squre, MPI_DOUBLE, bsend, 1, brecv, 1, comm3d, &status[1]);

  //multiply the block each processor owns
  serial_multiply(ablock, bblock, cblock, block_size);

  //initial futhur shifting ranks with cart_shift because step size and direction is now fixed for each shift of a and of b
  int up, down, left, right; 
  MPI_Cart_shift(comm3d, 0, 1, &up, &down);
  MPI_Cart_shift(comm3d, 1, 1, &left, &right);

  //shift and multiply until we had shifted round-1 times
  for(int round=0; round<round_shift-1; round++){
    MPI_Sendrecv_replace(ablock, block_squre, MPI_DOUBLE, right, 1, left, 1, comm3d, &status[0]);
    MPI_Sendrecv_replace(bblock, block_squre, MPI_DOUBLE, down, 1, up, 1, comm3d, &status[1]);

    serial_multiply(ablock, bblock, cblock, block_size);
    
  }

  //each Pxyz reducec block of c to Pxy0
  MPI_Reduce(cblock, reduced_cblock, block_squre, MPI_DOUBLE, MPI_SUM, 0, comm_z);
  
  if(rank_z==0){ 
    //P000 gather reduced blocks of c from Pxy0
    MPI_Gather(reduced_cblock, block_squre, MPI_DOUBLE, c, block_squre, MPI_DOUBLE, 0, comm_xy);
  }

  free(ablock);
  free(bblock);
  free(cblock);
  free(reduced_cblock);
}

//determine depth based on number of processors
int compute_depth(int cores){
  switch(cores){
    case 8: return 2;
    case 32: return 2;
    case 72: return 2;
    case 27: return 3;
    case 64: return 4;
    default: return 1;
  }
}

int main(int argc, char** argv) {

  //init mpi environment
  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  //get problem size and depth
  int n=atoi(argv[1]);
  int depth=compute_depth(world_size);

  //allocate memory for a b c
  double *a = (double *)malloc(n*n*sizeof(double));
  double *b = (double *)malloc(n*n*sizeof(double));
  double *c = (double *)malloc(n*n*sizeof(double));

  //time keeper
  double start, end;

  //only main processor P000 should populate matrix a and b
  if(world_rank==0){
    generateMatrix(a, b, n);
    start=MPI_Wtime();
  }

  //all processors works on 2.5d matrix multiply
  multiply_25d(a, b, c, n, depth);

  //main processor count time, print outputs and may also check reuslt.
  if(world_rank==0){
    end=MPI_Wtime();
    printf("Time %f\n", end-start);

    double dimension=std::sqrt(world_size/depth);
    double block_size=n/dimension;
    double round_shift=sqrt(world_size / pow(depth, 3));

    printf("n %d, p %d, c %d dimension %f block_size %f round_shift %f\n", n, world_size, depth, dimension, block_size, round_shift);
    //uncommand to check result
    //check_result(a, b, c,  n, block_size);
    
  }

  // Clean up
  free(a);
  free(b);
  free(c);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
