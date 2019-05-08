#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "utilities.h"
#include "Arduino.h"

#pragma once

const int num_layers = 3;
const int input_dims = 1;
const int max_memory = 20;
const int memory_batch_size = 5;

class Memory {
  public:
    int16_t **state_now;
    int16_t *action;
    int16_t **state_next;
    float **reward;
};

class QL_MODEL {
  public:
    int num_states;
    int num_actions;
    float ***Q;
    float **y;
    int16_t **goal_p;
    int epsilon;
    int max_epsilon;
    int min_epsilon;
    float learning_rate;
    float discount_rate; 
    float decay_rate;
    float reward_amplifier;

    /*BEGIN NEURAL NETWORK STUFF----------------------*/
    // Main variables
    const int16_t *layers;
    float **xl[num_layers-1];
    float **z[num_layers-1];
    float **weights[num_layers-1];
    float **biases[num_layers-1];
    enum ACTIVATION {
      TANH,
      SIGMOID,
      RELU
    };
    ACTIVATION activation_func;

    // Container variables
    int *max_action_idx;
    float **fp_H[num_layers];
    float **fp_w_T[num_layers-1];
    float **fp_mat1[num_layers-1];
    float **fp_mat2[num_layers-1];

    float **bp_diff;
    float **bp_diff_T;
    float **bp_grad_layer[num_layers-1];
    float **bp_w_delta[num_layers-1];
    float **bp_del[num_layers-1];
    float **bp_del_T[num_layers-1];
    float **bp_gradient[num_layers-1];
    float **bp_gradient_update_weights[num_layers-1];
    float **bp_gradient_update_biases[num_layers-1];
    /*END NEURAL NETWORK STUFF----------------------*/

    /*BEGIN CONSTRUCTORS & INITIALIZERS--------------*/
    QL_MODEL(QL_MODEL::ACTIVATION a, const int16_t *layers_p, int16_t **goal_p, int num_states, int num_actions, int min_epsilon, int max_epsilon, 
                                          float learning_rate, float discount_rate, float decay_rate, float reward_amplifier) {
      this->activation_func = a;
      this->layers = layers_p;
      this->goal_p = goal_p;
      this->num_states = num_states;
      this->num_actions = num_actions;
      this->min_epsilon = min_epsilon;
      this->max_epsilon = max_epsilon;
      this->epsilon = max_epsilon;
      this->learning_rate = learning_rate;
      this->discount_rate = discount_rate;
      this->decay_rate = decay_rate;
      this->reward_amplifier = reward_amplifier;
      this->allocate_params();
      this->xavier_init();
    }
    /*END CONSTRUCTORS & INITIALIZERS----------------*/

    int16_t* exploit(int16_t **x, int16_t *action);
    int16_t* explore(int16_t *action);
    void free_model(void);

    /*BEGIN NEURAL NETWORK STUFF----------------------*/
    void allocate_params();
    void xavier_init();
    void forward_pass(int16_t **x);
    void update_Q(int16_t **x, int16_t **x_new, float **reward);
    void back_propagate(int16_t **x);
    int* predict(int16_t **x);
    /*END NEURAL NETWORK STUFF----------------------*/

    /*BEGIN EXPERIENCE REPLAY STUFF------------------*/
    Memory mem[max_memory];
    int8_t memory_tracker_start;
    int8_t memory_tracker_stop;
    void insert_memory(int16_t **state_now, int16_t *action, int16_t **state_next, float **reward);
    void replay_random_experience(void);
    /*END EXPERIENCE REPLAY STUFF---------------------*/
};
