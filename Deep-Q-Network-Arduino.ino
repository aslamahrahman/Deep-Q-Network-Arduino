#include "q_learning.h"
#include "vars.h"

#pragma once

Matrix mat;
Utilities utl;

void initialize(void);
int16_t get_distance(void);
void get_state(int8_t when);
void drive(int16_t velocity);
void turn_right(int16_t velocity, int16_t pot_turn);
void turn_right(int16_t velocity, int16_t pot_turn);

void setup() {
  Serial.begin(115200);
  initialize();

  /*BEGIN LEDC STUFF--------------------------------------*/
  ledcSetup(led1_channel, freq, resolution);
  ledcAttachPin(pin_motor1_pwm, led1_channel);
  ledcSetup(led2_channel, freq, resolution);
  ledcAttachPin(pin_motor2_pwm, led2_channel);
  /*END LEDC STUFF---------------------------------------*/

  /*BEGIN Q LEARNING STUFF------------------------------*/
  state_now = utl.allocate_2D_int16_t(input_dims, 1);
  state_next = utl.allocate_2D_int16_t(input_dims, 1);
  diff = utl.allocate_2D_int16_t(input_dims, 1);
  action = utl.allocate_1D_int16_t(input_dims);
  reward = utl.allocate_2D_float(input_dims, 1);
  diff_reward = utl.allocate_2D_float(input_dims, 1);
  /*END Q LEARNING STUFF------------------------------*/

  Serial.println("Begin!");
  get_state(0);
}

void loop() {
  /*BEGIN Q LEARNING STUFF------------------------------*/
  counter++;
  Serial.printf("Counter: %d\n", counter);

  if(digitalRead(pin_training) == HIGH) {
    //Train
//    if(!mat.compare_mat(state_now, agent->goal_p, input_dims, 1)) { 
      //Choose
      int8_t random_number = random(0,100);
      if(random_number < agent->epsilon) {
        // Exploitaition
        #ifdef DEBUG
          Serial.print("Exploitation\n");
        #endif
        action = agent->exploit(state_now, action);
      }
      else {
        //Exploration
        #ifdef DEBUG
          Serial.print("Exploration\n");
        #endif
        action = agent->explore(action);
      }

      //Perform
      if(action[0] == 0) {
        drive(drive_speed);
      }
      else if(action[0] == 1) {
        drive(-drive_speed);
      }
      else {
        drive(0);
      }
      delay(1000);
      stop_now();

      #ifdef DEBUG
        Serial.printf("State now: %d\n", state_now[0][0]);
      #endif
      
      //Evaluate
      get_state(1);
//      if(mat.compare_mat(state_now, state_next, input_dims, 1)) {
        diff = mat.subtract_mat(diff, state_now, state_next, input_dims, 1); 
        diff_reward = mat.multiply_scalar(diff_reward, (float**)diff, agent->reward_amplifier, input_dims, 1);
        reward = mat.copy_to_existing(reward, diff_reward, input_dims, 1); 
        agent->insert_memory(state_now, action, state_next, reward);
        
        agent->forward_pass(state_now);
        if(counter % update_Q_rate == 0) {
          agent->update_Q(state_now, state_next, reward);
        }
        agent->back_propagate(state_now);
        
        agent->epsilon = min_epsilon + (max_epsilon - min_epsilon)*pow(2, -decay_rate*float(counter));
//      } 
      mat.copy_to_existing(state_now, state_next, input_dims, 1);

      if(counter % replay_rate == 0) {
        #ifdef DEBUG
          Serial.printf("Replaying experience\n");
        #endif
        agent->replay_random_experience();
      }
//    }
  }
  else {
    //Test
  }
  /*END Q LEARNING STUFF------------------------------*/
}

void initialize() {
  /*BEGIN TRAINING STUFF---------------------------------------*/
  pinMode(pin_training, INPUT);
  pinMode(pin_training_indicator, OUTPUT);
  /*END TRAINING STUFF-----------------------------------------*/
  
  /*BEGIN ULTRASONIC STUFF--------------------------------*/
  pinMode(pin_trigger, OUTPUT);
  pinMode(pin_echo, INPUT);
  /*END ULTRASONIC STUFF----------------------------------*/

  /*BEGIN MOTOR STUFF-------------------------------------*/
  pinMode(pin_motor1_pwm, OUTPUT);
  pinMode(pin_motor2_pwm, OUTPUT);
  pinMode(pin_motor1_dir1, OUTPUT);
  pinMode(pin_motor1_dir2, OUTPUT);
  pinMode(pin_motor2_dir1, OUTPUT);
  pinMode(pin_motor2_dir2, OUTPUT);
  /*END MOTOR STUFF-------------------------------------*/

  /*BEGIN Q LEARNING STUFF------------------------------*/
  agent = new QL_MODEL(QL_MODEL::ACTIVATION::TANH, layers_p, goal_p, num_states, num_actions, min_epsilon, max_epsilon, 
                        learning_rate, discount_rate, decay_rate, reward_amplifier);
  /*END Q LEARNING STUFF------------------------------*/
}

/*BEGIN ULTRASONIC STUFF--------------------------------*/
int16_t get_distance() {
  int16_t duration, cm = 0;
  int16_t num_samples = 20;
  for(int16_t i=0; i<num_samples; i++) {
    digitalWrite(pin_trigger, LOW);
    delayMicroseconds(2);
    digitalWrite(pin_trigger, HIGH);
    delayMicroseconds(10);
    digitalWrite(pin_trigger, LOW);
    delayMicroseconds(2);
    duration = pulseIn(pin_echo, HIGH, timeout);
    cm += (duration/29)/2;
  }
  cm = cm/num_samples;
  
  #ifdef DEBUG
    Serial.printf("Distance: %d\n", cm);
  #endif
  
  return cm;
}

void get_state(int8_t when) {
  int16_t dist = get_distance();

  if(when == 0) {
    if(dist<=100){
      state_now[0][0] = dist/10;
    }
    else {
      state_now[0][0] = num_states-1;
    }
  }
  else {
    if(dist<=100){
      state_next[0][0] = dist/10;
    }
    else {
      state_next[0][0] = num_states-1;
    }
  }
  return;
}
/*END ULTRASONIC STUFF----------------------------------*/

/*BEGIN MOTOR STUFF-------------------------------------*/
void drive(int16_t velocity) {
  if (velocity > drive_thresh_forward) {
    //go forward
    motor1_pwm = abs(velocity);
    motor2_pwm = abs(velocity);

    digitalWrite(pin_motor1_dir1, fw_dir1);
    digitalWrite(pin_motor1_dir2, fw_dir2);
    ledcWrite(led1_channel, motor1_pwm);

    digitalWrite(pin_motor2_dir1, fw_dir1);
    digitalWrite(pin_motor2_dir2, fw_dir2);
    ledcWrite(led2_channel, motor2_pwm);
  }
  else if (velocity < drive_thresh_backward) {
    //go backward
    motor1_pwm = abs(velocity);
    motor2_pwm = abs(velocity);

    digitalWrite(pin_motor1_dir1, bw_dir1);
    digitalWrite(pin_motor1_dir2, bw_dir2);
    ledcWrite(led1_channel, motor1_pwm);

    digitalWrite(pin_motor2_dir1, bw_dir1);
    digitalWrite(pin_motor2_dir2, bw_dir2);
    ledcWrite(led2_channel, motor2_pwm);
  }
  else {
    //stay right there
    motor1_pwm = 0;
    motor2_pwm = 0;
    ledcWrite(led1_channel, motor1_pwm);
    ledcWrite(led2_channel, motor2_pwm);
  }
}

void stop_now(void) {
  motor1_pwm = 0;
  motor2_pwm = 0;
  ledcWrite(led1_channel, motor1_pwm);
  ledcWrite(led2_channel, motor2_pwm);
}

void turn_left(int16_t velocity, int16_t pot_turn) {
  if (velocity > drive_thresh_forward) {
    //go forward left
    motor1_pwm = abs(velocity);
    motor2_pwm = is_turn_linear*(abs(velocity)*pot_turn/(3*turn_thresh) + 4*abs(velocity)/3);

    digitalWrite(pin_motor1_dir1, fw_dir1);
    digitalWrite(pin_motor1_dir2, fw_dir2);
    ledcWrite(led1_channel, motor1_pwm);

    digitalWrite(pin_motor2_dir1, fw_dir1);
    digitalWrite(pin_motor2_dir2, fw_dir2);
    ledcWrite(led2_channel, motor2_pwm);
  }
  else if (velocity < drive_thresh_backward) {
    //go backward left
    motor1_pwm = abs(velocity);
    motor2_pwm = is_turn_linear*(abs(velocity)*pot_turn/(3*turn_thresh) + 4*abs(velocity)/3);

    digitalWrite(pin_motor1_dir1, bw_dir1);
    digitalWrite(pin_motor1_dir2, bw_dir2);
    ledcWrite(led1_channel, motor1_pwm);

    digitalWrite(pin_motor2_dir1, bw_dir1);
    digitalWrite(pin_motor2_dir2, bw_dir2);
    ledcWrite(led2_channel, motor2_pwm);
  }
  else {
    //stay right there
    motor1_pwm = 0;
    motor2_pwm = 0;
    ledcWrite(led1_channel, motor1_pwm);
    ledcWrite(led2_channel, motor2_pwm);
  }
}

void turn_right(int16_t velocity, int16_t pot_turn) {
  if (velocity > drive_thresh_forward) {
    //go forward right
    motor1_pwm = is_turn_linear*(-abs(velocity)*pot_turn/(3*turn_thresh) + 4*abs(velocity)/3);
    motor2_pwm = abs(velocity);

    digitalWrite(pin_motor1_dir1, fw_dir1);
    digitalWrite(pin_motor1_dir2, fw_dir2);
    ledcWrite(led1_channel, motor1_pwm);

    digitalWrite(pin_motor2_dir1, fw_dir1);
    digitalWrite(pin_motor2_dir2, fw_dir2);
    ledcWrite(led2_channel, motor2_pwm);
  }
  else if (velocity < drive_thresh_backward) {
    //go backward right
    motor1_pwm = is_turn_linear*(-abs(velocity)*pot_turn/(3*turn_thresh) + 4*abs(velocity)/3);
    motor2_pwm = abs(velocity);
    
    digitalWrite(pin_motor1_dir1, bw_dir1);
    digitalWrite(pin_motor1_dir2, bw_dir2);
    ledcWrite(led1_channel, motor1_pwm);

    digitalWrite(pin_motor2_dir1, bw_dir1);
    digitalWrite(pin_motor2_dir2, bw_dir2);
    ledcWrite(led2_channel, motor2_pwm);
  }
  else {
    //stay right there
    motor1_pwm = 0;
    motor2_pwm = 0;
    ledcWrite(led1_channel, motor1_pwm);
    ledcWrite(led2_channel, motor2_pwm);
  }
}
/*END MOTOR STUFF---------------------------------------*/
