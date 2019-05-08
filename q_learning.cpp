#include "q_learning.h"

#pragma once

Utilities utlq;
Matrix matq;

void QL_MODEL::allocate_params() {
  int i=0;

  for(i=0; i<max_memory; i++) {
    this->mem[i].state_now = utlq.allocate_2D_int16_t(input_dims, 1);
    this->mem[i].action = utlq.allocate_1D_int16_t(input_dims);
    this->mem[i].state_next = utlq.allocate_2D_int16_t(input_dims, 1);
    this->mem[i].reward = utlq.allocate_2D_float(input_dims, 1);
  }
  this->memory_tracker_start = 0;
  this->memory_tracker_stop = 0;

  // Main variables
  for(i=0; i<num_layers-1; i++) {
    this->weights[i] = utlq.allocate_2D_float(this->layers[i], this->layers[i+1]);
    this->biases[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
    this->xl[i] = utlq.allocate_2D_float(this->layers[i], 1);
    this->z[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
  }

  this->Q = utlq.allocate_3D_float(this->num_states, this->num_actions, 1);
  this->y = utlq.allocate_2D_float(this->layers[num_layers-1], 1);

  // Container variables
  this->max_action_idx = utlq.allocate_1D_int(2);
  this->bp_diff = utlq.allocate_2D_float(this->layers[num_layers-1], 1);
  this->bp_diff_T = utlq.allocate_2D_float(1, this->layers[num_layers-1]);
  for(i=0; i<num_layers; i++) {
    this->fp_H[i] = utlq.allocate_2D_float(this->layers[i], 1);
  }
  for(i=0; i<num_layers-1; i++) {
    this->fp_w_T[i] = utlq.allocate_2D_float(this->layers[i+1], this->layers[i]);
    this->fp_mat1[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
    this->fp_mat2[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
    
    this->bp_grad_layer[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
    this->bp_w_delta[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
    this->bp_del[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
    this->bp_del_T[i] = utlq.allocate_2D_float(1, this->layers[i+1]);
    this->bp_gradient[i] = utlq.allocate_2D_float(this->layers[i], this->layers[i+1]);
    this->bp_gradient_update_weights[i] = utlq.allocate_2D_float(this->layers[i], this->layers[i+1]);
    this->bp_gradient_update_biases[i] = utlq.allocate_2D_float(this->layers[i+1], 1);
  }

  return;
}

void QL_MODEL::xavier_init() {
  int l, i, j;

  for(l=0; l<num_layers-1; l++) {
    for(i=0; i<this->layers[l]; i++) {
      for(j=0; j<this->layers[l+1]; j++) {
        float buf = sqrtf(6.0f/(this->layers[l] + this->layers[l+1]));
        this->weights[l][i][j] = -buf + 2.0f*buf*utlq.random_between_vals(this->layers[l], this->layers[l+1]);
      }
    }
    for(i=0; i<this->layers[l+1]; i++) {
      for(j=0; j<1; j++) {
        this->biases[l][i][j] = 0.0f;
      }
    }
  }
  return;
}

int16_t* QL_MODEL::explore(int16_t *action) {
  int i;
  for(i=0; i<input_dims; i++) {
    action[i] = rand()%3;
  }
  return action;
}

int16_t* QL_MODEL::exploit(int16_t **x, int16_t *action) {
  int i, j;
  this->max_action_idx = this->predict(x);
 
  for(i=0; i<input_dims; i++) {
    action[i] = max_action_idx[i];
  }
  
  return action;
}

void QL_MODEL::update_Q(int16_t **x, int16_t **x_new, float **reward) {
  float max_Q = 0.0f;
  int max_idx_action;
  int j;
  
  for(j=0; j<this->num_actions; j++) {
    if(this->Q[x_new[0][0]][j][0] > max_Q) {
      max_Q = this->Q[x_new[0][0]][j][0];
      max_idx_action = j;
    }
  }

  this->Q[x[0][0]][max_idx_action][0] = (1.0f - this->learning_rate)*this->Q[x[0][0]][max_idx_action][0] \
                                      + this->learning_rate*(reward[0][0] + this->discount_rate*max_Q);
  return;
}

void QL_MODEL::forward_pass(int16_t **x) {
  this->fp_H[0] = matq.copy_to_existing(this->fp_H[0], (float**)x, this->layers[0], 1);
  int l;

  if(this->activation_func == TANH) {
    for(l=0; l<num_layers-2; l++) {
      this->xl[l] = matq.copy_to_existing(this->xl[l], this->fp_H[l], this->layers[l], 1);
      this->fp_w_T[l] = matq.transpose(this->fp_w_T[l], this->weights[l], this->layers[l], this->layers[l+1]);
      this->fp_mat1[l] = matq.multiply_mat(this->fp_mat1[l], this->fp_w_T[l], this->layers[l+1], this->layers[l], this->fp_H[l], this->layers[l], 1);
      this->fp_mat2[l] = matq.add_mat(this->fp_mat2[l], this->fp_mat1[l], this->biases[l], this->layers[l+1], 1);
      this->z[l] = matq.copy_to_existing(this->z[l], this->fp_mat2[l], this->layers[l+1], 1);
      this->fp_H[l+1] = matq.tanh_mat(this->fp_H[l+1], this->fp_mat2[l], this->layers[l+1], 1);
    }
  }
  else if(this->activation_func == RELU) {
    for(l=0; l<num_layers-2; l++) {
      this->xl[l] = matq.copy_to_existing(this->xl[l], this->fp_H[l], this->layers[l], 1);
      this->fp_w_T[l] = matq.transpose(this->fp_w_T[l], this->weights[l], this->layers[l], this->layers[l+1]);
      this->fp_mat1[l] = matq.multiply_mat(this->fp_mat1[l], this->fp_w_T[l], this->layers[l+1], this->layers[l], this->fp_H[l], this->layers[l], 1);
      this->fp_mat2[l] = matq.add_mat(this->fp_mat2[l], this->fp_mat1[l], this->biases[l], this->layers[l+1], 1);
      this->z[l] = matq.copy_to_existing(this->z[l], this->fp_mat2[l], this->layers[l+1], 1);
      this->fp_H[l+1] = matq.relu_mat(this->fp_H[l+1], this->fp_mat2[l], this->layers[l+1], 1);
    }
  }

  // Linear layer
  this->xl[l] = matq.copy_to_existing(this->xl[l], this->fp_H[l], this->layers[l], 1);
  this->fp_w_T[l] = matq.transpose(this->fp_w_T[l], this->weights[l], this->layers[l], this->layers[l+1]);
  this->fp_mat1[l] = matq.multiply_mat(this->fp_mat1[l], this->fp_w_T[l], this->layers[l+1], this->layers[l], this->fp_H[l], this->layers[l], 1);
  this->fp_H[l+1] = matq.add_mat(this->fp_H[l+1], this->fp_mat1[l], this->biases[l], this->layers[l+1], 1);
  this->z[l] = matq.copy_to_existing(this->z[l], this->fp_H[l+1], this->layers[l+1], 1);
  this->y = matq.copy_to_existing(this->y, this->z[l], this->layers[l+1], 1);
  
  return;
}

void QL_MODEL::back_propagate(int16_t **x) {
  int l;

  if(this->activation_func == TANH) {
    this->bp_diff = matq.subtract_mat(this->bp_diff, this->y, this->Q[x[0][0]], this->layers[num_layers-1], 1);
    this->bp_diff_T = matq.transpose(this->bp_diff_T, this->bp_diff, this->layers[num_layers-1], 1);
    this->bp_grad_layer[num_layers-2] = matq.ones_mat(this->bp_grad_layer[num_layers-2], this->layers[num_layers-1], 1);  // needs correction
    this->bp_del[num_layers-2] = matq.multiply_element_mat(this->bp_del[num_layers-2], this->bp_diff, this->bp_grad_layer[num_layers-2], this->layers[num_layers-1], 1); 
    this->bp_del_T[num_layers-2] = matq.transpose(this->bp_del_T[num_layers-2], this->bp_del[num_layers-2], this->layers[num_layers-1], 1);
    this->bp_gradient[num_layers-2] = matq.multiply_mat(this->bp_gradient[num_layers-2], this->xl[num_layers-2], this->layers[num_layers-2], 1, 
                                      this->bp_del_T[num_layers-2], 1, this->layers[num_layers-1]);
    this->bp_gradient_update_weights[num_layers-2] = matq.multiply_scalar(this->bp_gradient_update_weights[num_layers-2], this->bp_gradient[num_layers-2],
                                                      this->learning_rate, this->layers[num_layers-2], this->layers[num_layers-1]);
    this->bp_gradient_update_biases[num_layers-2] = matq.multiply_scalar(this->bp_gradient_update_biases[num_layers-2], this->bp_del[num_layers-2], 
                                                      this->learning_rate, this->layers[num_layers-1], 1);
    this->weights[num_layers-2] = matq.subtract_mat(this->weights[num_layers-2], this->weights[num_layers-2], this->bp_gradient_update_weights[num_layers-2], 
                                  this->layers[num_layers-2], this->layers[num_layers-1]);
    this->biases[num_layers-2] = matq.subtract_mat(this->biases[num_layers-2], this->biases[num_layers-2], this->bp_gradient_update_biases[num_layers-2], 
                                  this->layers[num_layers-1], 1);
    
    for(l=num_layers-3; l>=0; l--) {
      this->bp_grad_layer[l] = matq.grad_tanh_mat(this->bp_grad_layer[l], this->z[l], this->layers[l+1], 1);
      this->bp_w_delta[l] = matq.multiply_mat(this->bp_w_delta[l], this->weights[l+1], this->layers[l+1], this->layers[l+2], this->bp_del[l+1], this->layers[l+2], 1);
      this->bp_del[l] = matq.multiply_element_mat(this->bp_del[l], this->bp_w_delta[l], this->bp_grad_layer[l], this->layers[l+1], 1);
      this->bp_del_T[l] = matq.transpose(this->bp_del_T[l], this->bp_del[l], this->layers[l+1], 1);
      this->bp_gradient[l] = matq.multiply_mat(this->bp_gradient[l], this->xl[l], this->layers[l], 1, this->bp_del_T[l], 1, this->layers[l+1]);
      this->bp_gradient_update_weights[l] = matq.multiply_scalar(this->bp_gradient_update_weights[l], this->bp_gradient[l], this->learning_rate, this->layers[l], this->layers[l+1]);
      this->bp_gradient_update_biases[l] = matq.multiply_scalar(this->bp_gradient_update_biases[l], this->bp_del[l], this->learning_rate, this->layers[l+1], 1);
      this->weights[l] = matq.subtract_mat(this->weights[l], this->weights[l], this->bp_gradient_update_weights[l], this->layers[l], this->layers[l+1]);
      this->biases[l] = matq.subtract_mat(this->biases[l], this->biases[l], this->bp_gradient_update_biases[l], this->layers[l+1], 1);
    }
  }
  else if(this->activation_func == RELU) {
    this->bp_diff = matq.subtract_mat(this->bp_diff, this->y, this->Q[x[0][0]], this->layers[num_layers-1], 1);
    this->bp_diff_T = matq.transpose(this->bp_diff_T, this->bp_diff, this->layers[num_layers-1], 1);
    this->bp_grad_layer[num_layers-2] = matq.ones_mat(this->bp_grad_layer[num_layers-2], this->layers[num_layers-1], 1);  // needs correction
    this->bp_del[num_layers-2] = matq.multiply_element_mat(this->bp_del[num_layers-2], this->bp_diff, this->bp_grad_layer[num_layers-2], this->layers[num_layers-1], 1); 
    this->bp_del_T[num_layers-2] = matq.transpose(this->bp_del_T[num_layers-2], this->bp_del[num_layers-2], this->layers[num_layers-1], 1);
    this->bp_gradient[num_layers-2] = matq.multiply_mat(this->bp_gradient[num_layers-2], this->xl[num_layers-2], this->layers[num_layers-2], 1, 
                                      this->bp_del_T[num_layers-2], 1, this->layers[num_layers-1]);
    this->bp_gradient_update_weights[num_layers-2] = matq.multiply_scalar(this->bp_gradient_update_weights[num_layers-2], this->bp_gradient[num_layers-2],
                                                      this->learning_rate, this->layers[num_layers-2], this->layers[num_layers-1]);
    this->bp_gradient_update_biases[num_layers-2] = matq.multiply_scalar(this->bp_gradient_update_biases[num_layers-2], this->bp_del[num_layers-2], 
                                                      this->learning_rate, this->layers[num_layers-1], 1);
    this->weights[num_layers-2] = matq.subtract_mat(this->weights[num_layers-2], this->weights[num_layers-2], this->bp_gradient_update_weights[num_layers-2], 
                                  this->layers[num_layers-2], this->layers[num_layers-1]);
    this->biases[num_layers-2] = matq.subtract_mat(this->biases[num_layers-2], this->biases[num_layers-2], this->bp_gradient_update_biases[num_layers-2], 
                                  this->layers[num_layers-1], 1);
    
    for(l=num_layers-3; l>=0; l--) {
      this->bp_grad_layer[l] = matq.grad_relu_mat(this->bp_grad_layer[l], this->z[l], this->layers[l+1], 1);
      this->bp_w_delta[l] = matq.multiply_mat(this->bp_w_delta[l], this->weights[l+1], this->layers[l+1], this->layers[l+2], this->bp_del[l+1], this->layers[l+2], 1);
      this->bp_del[l] = matq.multiply_element_mat(this->bp_del[l], this->bp_w_delta[l], this->bp_grad_layer[l], this->layers[l+1], 1);
      this->bp_del_T[l] = matq.transpose(this->bp_del_T[l], this->bp_del[l], this->layers[l+1], 1);
      this->bp_gradient[l] = matq.multiply_mat(this->bp_gradient[l], this->xl[l], this->layers[l], 1, this->bp_del_T[l], 1, this->layers[l+1]);
      this->bp_gradient_update_weights[l] = matq.multiply_scalar(this->bp_gradient_update_weights[l], this->bp_gradient[l], this->learning_rate, this->layers[l], this->layers[l+1]);
      this->bp_gradient_update_biases[l] = matq.multiply_scalar(this->bp_gradient_update_biases[l], this->bp_del[l], this->learning_rate, this->layers[l+1], 1);
      this->weights[l] = matq.subtract_mat(this->weights[l], this->weights[l], this->bp_gradient_update_weights[l], this->layers[l], this->layers[l+1]);
      this->biases[l] = matq.subtract_mat(this->biases[l], this->biases[l], this->bp_gradient_update_biases[l], this->layers[l+1], 1);
    }
  }
  
  return;
}

int* QL_MODEL::predict(int16_t **x) {
  this->forward_pass(x);
  this->max_action_idx = matq.max_idx_mat(this->max_action_idx, this->y, this->layers[num_layers-1], 1);
  return this->max_action_idx; //needs correction
}

void QL_MODEL::free_model() {
  delete(this->Q); this->Q = NULL;
  delete(this->y); this->y = NULL;

  int i;
  for(i=0; i<num_layers-1; i++) {
    delete(this->weights[i]); this->weights[i] = NULL;
    delete(this->biases[i]); this->biases[i] = NULL;
    delete(this->xl[i]); this->xl[i] = NULL;
    delete(this->z[i]); this->z[i] = NULL;
  }

  delete(this->max_action_idx); this->max_action_idx = NULL;
  delete(this->bp_diff); this->bp_diff = NULL;
  delete(this->bp_diff_T); this->bp_diff_T = NULL;
  for(i=0; i<num_layers; i++) {
    delete(this->fp_H[i]); this->fp_H[i] = NULL;
  }
  for(i=0; i<num_layers-1; i++) {
    delete(this->fp_w_T[i]); this->fp_w_T[i] = NULL;
    delete(this->fp_mat1[i]); this->fp_mat1[i] = NULL;
    delete(this->fp_mat2[i]); this->fp_mat2[i] = NULL;
    
    delete(this->bp_grad_layer[i]); this->bp_grad_layer[i] = NULL;
    delete(this->bp_w_delta[i]); this->bp_w_delta[i] = NULL;
    delete(this->bp_del[i]); this->bp_del[i] = NULL;
    delete(this->bp_del_T[i]); this->bp_del_T[i] = NULL;
    delete(this->bp_gradient[i]); this->bp_gradient[i] = NULL;
    delete(this->bp_gradient_update_weights[i]); this->bp_gradient_update_weights[i] = NULL;
    delete(this->bp_gradient_update_biases[i]); bp_gradient_update_biases[i] = NULL;
  }

  return;
}
