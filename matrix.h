#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utilities.h"
#include "Arduino.h"

#pragma once

class Matrix {
  public:
    float* int_to_float(int *m, int nx);
    float** int_to_float(int **m, int nx, int ny);
    float*** int_to_float(int ***m, int nx, int ny, int nz);
    float* copy(float *m, int nx);
    float** copy(float **m, int nx, int ny);
    float*** copy(float ***m, int nx, int ny, int nz);
    int* copy_to_existing(int *m1, int *m2, int nx);
    int** copy_to_existing(int **m1, int **m2, int nx, int ny);
    int*** copy_to_existing(int ***m1, int ***m2, int nx, int ny, int nz);
    int16_t* copy_to_existing(int16_t *m1, int16_t *m2, int nx);
    int16_t** copy_to_existing(int16_t **m1, int16_t **m2, int nx, int ny);
    int16_t*** copy_to_existing(int16_t ***m1, int16_t ***m2, int nx, int ny, int nz);
    float* copy_to_existing(float *m1, float *m2, int nx);
    float** copy_to_existing(float **m1, float **m2, int nx, int ny);
    float*** copy_to_existing(float ***m1, float ***m2, int nx, int ny, int nz);
    bool compare_mat(int *m1, int *m2, int nx);
    bool compare_mat(int **m1, int **m2, int nx, int ny);
    bool compare_mat(int ***m1, int ***m2, int nx, int ny, int nz);
    bool compare_mat(int16_t *m1, int16_t *m2, int nx);
    bool compare_mat(int16_t **m1, int16_t **m2, int nx, int ny);
    bool compare_mat(int16_t ***m1, int16_t ***m2, int nx, int ny, int nz);
    float* sub_mat(float *m, int nx1, int nx2);
    float** sub_mat(float **m, int nx1, int nx2, int ny1, int ny2);
    float*** sub_mat(float ***m, int nx1, int nx2, int ny1, int ny2, int nz1, int nz2);
    float* transpose(float *m_T, float *m, int nx);
    float** transpose(float **m_T, float **m, int nx, int ny);
    float* add_scalar(float *m, float val, int nx);
    float** add_scalar(float **m, float val, int nx, int ny);
    float*** add_scalar(float ***m, float val, int nx, int ny, int nz);
    float* add_mat(float *sum, float *m1, float *m2, int nx);
    float** add_mat(float **sum, float **m1, float **m2, int nx, int ny);
    float*** add_mat(float ***sum, float ***m1, float ***m2, int nx, int ny, int nz);
    int* subtract_mat(int *diff, int *m1, int *m2, int nx);
    int** subtract_mat(int **diff, int **m1, int **m2, int nx, int ny);
    int*** subtract_mat(int ***diff, int ***m1, int ***m2, int nx, int ny, int nz);
    int16_t* subtract_mat(int16_t *diff, int16_t *m1, int16_t *m2, int nx);
    int16_t** subtract_mat(int16_t **diff, int16_t **m1, int16_t **m2, int nx, int ny);
    int16_t*** subtract_mat(int16_t ***diff, int16_t ***m1, int16_t ***m2, int nx, int ny, int nz);
    float* subtract_mat(float *diff, float *m1, float *m2, int nx);
    float** subtract_mat(float **diff, float **m1, float **m2, int nx, int ny);
    float*** subtract_mat(float ***diff, float ***m1, float ***m2, int nx, int ny, int nz);
    float* multiply_scalar(float *mul, float *m, float val, int nx);
    float** multiply_scalar(float **mul, float **m, float val, int nx, int ny);
    float*** multiply_scalar(float ***mul, float ***m, float val, int nx, int ny, int nz);
    float* multiply_element_mat(float *mul, float *m1, float *m2, int nx);
    float** multiply_element_mat(float **mul, float **m1, float **m2, int nx, int ny);
    float*** multiply_element_mat(float ***mul, float ***m1, float ***m2, int nx, int ny, int nz);
    float dot_mat(float *m1, float *m2, int nx);
    float* dot_mat(float *dot, float **m1, float **m2, int nx, int ny);
    float** dot_mat(float **dot, float ***m1, float ***m2, int nx, int ny, int nz);
    float** multiply_mat(float **matmul, float **m1, int nx1, int ny1, float **m2, int nx2, int ny2);
    float* divide_scalar(float *m, float val, int nx);
    float** divide_scalar(float **m, float val, int nx, int ny);
    float*** divide_scalar(float ***m, float val, int nx, int ny, int nz);
    float* divide_mat(float *divs, float *m1, float *m2, int nx);
    float** divide_mat(float **divs, float **m1, float **m2, int nx, int ny);
    float*** divide_mat(float ***divs, float ***m1, float ***m2, int nx, int ny, int nz);
    float* tanh_mat(float *tanhmat, float *m, int nx);
    float** tanh_mat(float **tanhmat, float **m, int nx, int ny);
    float*** tanh_mat(float ***tanhmat, float ***m, int nx, int ny, int nz);
    float* grad_tanh_mat(float *gtanh, float *m, int nx);
    float** grad_tanh_mat(float **gtanh, float **m, int nx, int ny);
    float*** grad_tanh_mat(float ***gtanh, float ***m, int nx, int ny, int nz);
    float* relu_mat(float *relumat, float *m, int nx);
    float** relu_mat(float **relumat, float **m, int nx, int ny);
    float*** relu_mat(float ***relumat, float ***m, int nx, int ny, int nz);
    float* grad_relu_mat(float *grelu, float *m, int nx);
    float** grad_relu_mat(float **grelu, float **m, int nx, int ny);
    float*** grad_relu_mat(float ***grelu, float ***m, int nx, int ny, int nz);
    float* sigmoid_mat(float *sigmoidmat, float *m, int nx);
    float** sigmoid_mat(float **sigmoidmat, float **m, int nx, int ny);
    float*** sigmoid_mat(float ***sigmoidmat, float ***m, int nx, int ny, int nz);
    float* grad_sigmoid_mat(float *gsigmoid, float *m, int nx);
    float** grad_sigmoid_mat(float **gsigmoid, float **m, int nx, int ny);
    float*** grad_sigmoid_mat(float ***gsigmoid, float ***m, int nx, int ny, int nz);
    float l2norm_mat(float *m, int nx);
    float l2norm_mat(float **m, int nx, int ny);
    float l2norm_mat(float ***m, int nx, int ny, int nz);
    float max_mat(float *m, int nx);
    float max_mat(float **m, int nx, int ny);
    float max_mat(float ***m, int nx, int ny, int nz);
    int max_idx_mat(float *m, int nx);
    int* max_idx_mat(int *idx, float **m, int nx, int ny);
    int* max_idx_mat(int *idx, float ***m, int nx, int ny, int nz);
    void print_mat(int *m, int nx);
    void print_mat(int **m, int nx, int ny);
    void print_mat(int ***m, int nx, int ny, int nz);
    void print_mat(float *m, int nx);
    void print_mat(float **m, int nx, int ny);
    void print_mat(float ***m, int nx, int ny, int nz);
    float* ones_mat(float *ones, int nx);
    float** ones_mat(float **ones, int nx, int ny);
    float*** ones_mat(float ***ones, int nx, int ny, int nz);
};