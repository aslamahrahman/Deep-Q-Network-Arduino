#include "q_learning.h"

#pragma once

Utilities utlqm;
Matrix matqm;

void QL_MODEL::insert_memory(int16_t **state_now, int16_t *action, int16_t **state_next, float **reward) {
  int start = this->memory_tracker_start;
  int stopp = this->memory_tracker_stop;
  int diff = start - stopp;
  if(diff <= 0) {
    if(abs(diff) < (max_memory - 1)) {
      matqm.copy_to_existing(this->mem[stopp].state_now, state_now, input_dims, 1);
      matqm.copy_to_existing(this->mem[stopp].action, action, input_dims);
      matqm.copy_to_existing(this->mem[stopp].state_next, state_next, input_dims, 1);
      matqm.copy_to_existing(this->mem[stopp].reward, reward, input_dims, 1);
      this->memory_tracker_stop += 1;
    }
    else {
      matqm.copy_to_existing(this->mem[start].state_now, state_now, input_dims, 1);
      matqm.copy_to_existing(this->mem[start].action, action, input_dims);
      matqm.copy_to_existing(this->mem[start].state_next, state_next, input_dims, 1);
      matqm.copy_to_existing(this->mem[start].reward, reward, input_dims, 1);
      this->memory_tracker_start += 1;
      this->memory_tracker_stop = 0;
    }
  }
  else {
    if(start != (max_memory-1)) {
      matqm.copy_to_existing(this->mem[start].state_now, state_now, input_dims, 1);
      matqm.copy_to_existing(this->mem[start].action, action, input_dims);
      matqm.copy_to_existing(this->mem[start].state_next, state_next, input_dims, 1);
      matqm.copy_to_existing(this->mem[start].reward, reward, input_dims, 1);
      this->memory_tracker_start += 1;
      this->memory_tracker_stop += 1;
    }
    else {
      matqm.copy_to_existing(this->mem[0].state_now, state_now, input_dims, 1);
      matqm.copy_to_existing(this->mem[0].action, action, input_dims);
      matqm.copy_to_existing(this->mem[0].state_next, state_next, input_dims, 1);
      matqm.copy_to_existing(this->mem[0].reward, reward, input_dims, 1);
      this->memory_tracker_start = 0;
      this->memory_tracker_stop += 1;
    }
  }

  return;
}

void QL_MODEL::replay_random_experience() {
  int16_t random_number;

  for(int b=0; b<memory_batch_size; b++) {
    if(this->memory_tracker_stop < (max_memory)) {
      random_number = random(0,this->memory_tracker_stop);
    }
    else {
      random_number = random(0,max_memory);
    }
  
    this->forward_pass(this->mem[random_number].state_now);
    
    if(!matqm.compare_mat(this->mem[random_number].state_now, this->goal_p, input_dims, 1)) { 
      float max_Q = 0.0f;
      int max_idx_action;
      int j;
    
      for(j=0; j<this->num_actions; j++) {
        if(this->Q[this->mem[random_number].state_next[0][0]][j][0] > max_Q) {
          max_Q = this->Q[this->mem[random_number].state_next[0][0]][j][0];
          max_idx_action = j;
        }
      }
  
      this->y[this->mem[random_number].action[0]][0] =  this->mem[random_number].reward[0][0] + this->learning_rate*(this->discount_rate*max_Q);
    }
    else {
      this->y[this->mem[random_number].action[0]][0] = this->mem[random_number].reward[0][0];
    }
    this->back_propagate(this->mem[random_number].state_now);
  }

  return;
}
