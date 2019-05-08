/*BEGIN CONTROLS STUFF-----------------------------------------*/
#define LED_BUILTIN 2
#define DEBUG true
/*END CONTROLS STUFF--------------------------------------------*/

/*BEGIN TRAINING STUFF---------------------------------------*/
const int pin_training = 36;
const int pin_training_indicator = 22;
/*END TRAINING STUFF-----------------------------------------*/
  
/*BEGIN ULTRASONIC STUFF--------------------------------*/
const int pin_trigger = 25;
const int pin_echo = 26;
const int timeout = 30000;
/*END ULTRASONIC STUFF----------------------------------*/

/*BEGIN MOTOR STUFF-------------------------------------*/
//define motor variables
int motor1_pwm = 0;
int motor2_pwm = 0;
int motor1_dir1 = 0;
int motor1_dir2 = 0;
int motor2_dir1 = 0;
int motor2_dir2 = 0;

const int pin_motor1_pwm = 23;
const int pin_motor2_pwm = 22;
const int pin_motor1_dir1 = 19;
const int pin_motor1_dir2 = 18;
const int pin_motor2_dir1 = 17;
const int pin_motor2_dir2 = 16;

//constant speed for stage 1
const int drive_speed = 100;

//define thresholds to remove noise from pot readings
const int drive_thresh = 50;
const int drive_thresh_forward = drive_thresh;
const int drive_thresh_backward = -drive_thresh;
const int turn_thresh = 25;
const int dir_thresh_right = turn_thresh;
const int dir_thresh_left = -turn_thresh;

//direction pins values, caster wheel on left, motor 2 nearer to you, forward is anti-clockwise rotation, backward is clockwise rotation
const bool bw_dir1 = false;
const bool bw_dir2 = true;
const bool fw_dir1 = true;
const bool fw_dir2 = false;
const int is_turn_linear = 1;
/*END MOTOR STUFF---------------------------------------*/

/*BEGIN LEDC STUFF----------------------------------------*/
const int freq = 1200;
const int led1_channel = 0;
const int led2_channel = 1;
const int resolution = 8;
/*END LEDC STUFF-------------------------------------------*/

/*BEGIN Q LEARNING STUFF------------------------------*/
QL_MODEL *agent;
const int num_actions = 3;
const int num_states = 10;
int16_t goal[1][1] = {{1}};
int16_t *goal_pm[1] = {goal[0]};
int16_t **goal_p = goal_pm;
int max_epsilon = 80;
int min_epsilon = 1;
int decay_rate = 0.001f;
float learning_rate = 0.1f;
float discount_rate = 0.9f; 
float reward_amplifier = 1.0f;
int update_Q_rate = 1;
int replay_rate = 1;

const int16_t layers[] = {input_dims, 2, num_actions};
const int16_t *layers_p = layers;

int16_t **state_now, **state_next, *action, **diff;
float **reward, **diff_reward;
long counter = 0;
/*END Q LEARNING STUFF--------------------------------*/
