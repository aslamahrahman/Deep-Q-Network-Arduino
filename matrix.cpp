#include "matrix.h"

Utilities utlm;

float* Matrix::copy(float *m, int nx) {
  int i;
  float* cop = utlm.allocate_1D_float(nx);
  for(i=0; i<nx; i++) {
    cop[i] = m[i];
  }
  return cop;
}

float** Matrix::copy(float **m, int nx, int ny) {
  int i, j;
  float** cop = utlm.allocate_2D_float(nx, ny);
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      cop[i][j] = m[i][j];
    }
  }
  return cop;
}

float*** Matrix::copy(float ***m, int nx, int ny, int nz) {
  int i, j, k;
  float*** cop = utlm.allocate_3D_float(nx, ny, nz);
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        cop[i][j][k] = m[i][j][k];
      }
    }
  }
  return cop;
}

int* Matrix::copy_to_existing(int *m1, int *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    m1[i] = m2[i];
  }
  return m1;
}

int** Matrix::copy_to_existing(int **m1, int **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      m1[i][j] = m2[i][j];
    }
  }
  return m1;
}

int*** Matrix::copy_to_existing(int ***m1, int ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        m1[i][j][k] = m2[i][j][k];
      }
    }
  }
  return m1;
}

int16_t* Matrix::copy_to_existing(int16_t *m1, int16_t *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    m1[i] = m2[i];
  }
  return m1;
}

int16_t** Matrix::copy_to_existing(int16_t **m1, int16_t **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      m1[i][j] = m2[i][j];
    }
  }
  return m1;
}

int16_t*** Matrix::copy_to_existing(int16_t ***m1, int16_t ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        m1[i][j][k] = m2[i][j][k];
      }
    }
  }
  return m1;
}

float* Matrix::copy_to_existing(float *m1, float *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    m1[i] = m2[i];
  }
  return m1;
}

float** Matrix::copy_to_existing(float **m1, float **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      m1[i][j] = m2[i][j];
    }
  }
  return m1;
}

float*** Matrix::copy_to_existing(float ***m1, float ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        m1[i][j][k] = m2[i][j][k];
      }
    }
  }
  return m1;
}

bool Matrix::compare_mat(int *m1, int *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    if(m1[i] != m2[i]) {
      return false;
    }
  }
  return true;
}

bool Matrix::compare_mat(int **m1, int **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      if(m1[i][j] != m2[i][j]) {
        return false;
      }
    }
  }
  return true;
}

bool Matrix::compare_mat(int ***m1, int ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        if(m1[i][j][k] != m2[i][j][k]) {
          return false;
        }
      }
    }
  }
  return true;
}

bool Matrix::compare_mat(int16_t *m1, int16_t *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    if(m1[i] != m2[i]) {
      return false;
    }
  }
  return true;
}

bool Matrix::compare_mat(int16_t **m1, int16_t **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      if(m1[i][j] != m2[i][j]) {
        return false;
      }
    }
  }
  return true;
}

bool Matrix::compare_mat(int16_t ***m1, int16_t ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        if(m1[i][j][k] != m2[i][j][k]) {
          return false;
        }
      }
    }
  }
  return true;
}

float* Matrix::transpose(float *m_T, float *m, int nx) {
	int i;
	for(i=0; i<nx; i++) {
		m_T[i] = m[nx-i-1];
	}
	return m_T;
}

float** Matrix::transpose(float **m_T, float **m, int nx, int ny) {
	int i, j;
	for(i=0; i<ny; i++) {
		for(j=0; j<nx; j++) {
			m_T[i][j] = m[j][i];
		}
	}
	return m_T;
}

float* Matrix::add_scalar(float *m, float val, int nx) {
	int i;
	for(i=0; i<nx; i++) {
		m[i] += val;
	}
	return m;
}

float** Matrix::add_scalar(float **m, float val, int nx, int ny) {
	int i, j;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			m[i][j] += val;
		}
	}
	return m;
}

float*** Matrix::add_scalar(float ***m, float val, int nx, int ny, int nz) {
	int i, j, k;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			for(k=0; k<nz; k++) {
				m[i][j][k] += val;
			}
		}
	}
	return m;
}

float* Matrix::add_mat(float *sum, float *m1, float *m2, int nx) {
	int i;
	for(i=0; i<nx; i++) {
		sum[i] = m1[i] + m2[i];
	}
	return sum;
}

float** Matrix::add_mat(float **sum, float **m1, float **m2, int nx, int ny) {
	int i, j;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			sum[i][j] = m1[i][j] + m2[i][j];
		}
	}
	return sum;
}

float*** Matrix::add_mat(float ***sum, float ***m1, float ***m2, int nx, int ny, int nz) {
	int i, j, k;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			for(k=0; k<nz; k++) {
				sum[i][j][k] = m1[i][j][k] + m2[i][j][k];
			}
		}
	}
	return sum;
}

int* Matrix::subtract_mat(int *diff, int *m1, int *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    diff[i] = m1[i] - m2[i];
  }
  return diff;
}

int** Matrix::subtract_mat(int **diff, int **m1, int **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      diff[i][j] = m1[i][j] - m2[i][j];
    }
  }
  return diff;
}

int*** Matrix::subtract_mat(int ***diff, int ***m1, int ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        diff[i][j][k] = m1[i][j][k] - m2[i][j][k];
      }
    }
  }
  return diff;
}

int16_t* Matrix::subtract_mat(int16_t *diff, int16_t *m1, int16_t *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    diff[i] = m1[i] - m2[i];
  }
  return diff;
}

int16_t** Matrix::subtract_mat(int16_t **diff, int16_t **m1, int16_t **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      diff[i][j] = m1[i][j] - m2[i][j];
    }
  }
  return diff;
}

int16_t*** Matrix::subtract_mat(int16_t ***diff, int16_t ***m1, int16_t ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        diff[i][j][k] = m1[i][j][k] - m2[i][j][k];
      }
    }
  }
  return diff;
}

float* Matrix::subtract_mat(float *diff, float *m1, float *m2, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    diff[i] = m1[i] - m2[i];
  }
  return diff;
}

float** Matrix::subtract_mat(float **diff, float **m1, float **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      diff[i][j] = m1[i][j] - m2[i][j];
    }
  }
  return diff;
}

float*** Matrix::subtract_mat(float ***diff, float ***m1, float ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        diff[i][j][k] = m1[i][j][k] - m2[i][j][k];
      }
    }
  }
  return diff;
}

float* Matrix::multiply_scalar(float *mul, float *m, float val, int nx) {
	int i;
	for(i=0; i<nx; i++) {
		mul[i] = m[i] * val;
	}
	return mul;
}

float** Matrix::multiply_scalar(float **mul, float **m, float val, int nx, int ny) {
	int i, j;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			mul[i][j] = m[i][j] * val;
		}
	}
	return mul;
}

float*** Matrix::multiply_scalar(float ***mul, float ***m, float val, int nx, int ny, int nz) {
	int i, j, k;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			for(k=0; k<nz; k++) {
				mul[i][j][k] = m[i][j][k] * val;
			}
		}
	}
	return mul;
}

float Matrix::dot_mat(float *m1, float *m2, int nx) {
  int i;
  float dot_product = 0;
  for(i=0; i<nx; i++) {
    dot_product += m1[i] * m2[i];
  }
  return dot_product;
}

float* Matrix::dot_mat(float *dot, float **m1, float **m2, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      dot[i] += m1[i][j] * m2[i][j];
    }
  }
  return dot;
}

float** Matrix::dot_mat(float **dot, float ***m1, float ***m2, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        dot[i][j] += m1[i][j][k] * m2[i][j][k];
      }
    }
  }
  return dot;
}

float* Matrix::multiply_element_mat(float *mul, float *m1, float *m2, int nx) {
	int i;
	for(i=0; i<nx; i++) {
		mul[i] = m1[i] * m2[i];
	}
	return mul;
}

float** Matrix::multiply_element_mat(float **mul, float **m1, float **m2, int nx, int ny) {
	int i, j;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			mul[i][j] = m1[i][j] * m2[i][j];
		}
	}
	return mul;
}

float*** Matrix::multiply_element_mat(float ***mul, float ***m1, float ***m2, int nx, int ny, int nz) {
	int i, j, k;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			for(k=0; k<nz; k++) {
				mul[i][j][k] = m1[i][j][k] * m2[i][j][k];
			}
		}
	}
	return mul;
}

float** Matrix::multiply_mat(float **matmul, float **m1, int nx1, int ny1, float **m2, int nx2, int ny2) {
	int i, j, k;
	if(ny1 == nx2) {
		for(i=0; i<nx1; i++) {
			for(j=0; j<ny2; j++) {
				matmul[i][j] = 0.0f;
				for(k=0; k<ny1; k++) {
					matmul[i][j] += m1[i][k]*m2[k][j];
				}
			}
		}
	}
  else {
    return NULL;
  }
	return matmul;
}

float* Matrix::divide_scalar(float *m, float val, int nx) {
	int i;
  float	val_by_1 = 1.0f/val;
	for(i=0; i<nx; i++) {
		 m[i] *= val_by_1;
	}
	return m;
}

float** Matrix::divide_scalar(float **m, float val, int nx, int ny) {
	int i, j;
  float	val_by_1 = 1.0/val;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			m[i][j] *= val_by_1;
		}
	}
	return m;
}

float*** Matrix::divide_scalar(float ***m, float val, int nx, int ny, int nz) {
	int i, j, k;
  float	val_by_1 = 1.0/val;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			for(k=0; k<nz; k++) {
				m[i][j][k] *= val_by_1;
			}
		}
	}
	return m;
}

float* Matrix::tanh_mat(float *tanhmat, float *m, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    tanhmat[i] = tanhf(m[i]);
  }
  return tanhmat;
}

float** Matrix::tanh_mat(float **tanhmat, float **m, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      tanhmat[i][j] = tanhf(m[i][j]);
    }
  }
  return tanhmat;
}

float*** Matrix::tanh_mat(float ***tanhmat, float ***m, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        tanhmat[i][j][k] = tanhf(m[i][j][k]);
      }
    }
  }
  return tanhmat;
}

float* Matrix::grad_tanh_mat(float *gtanh, float *m, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    gtanh[i] = 1.0f - tanhf(m[i])*tanhf(m[i]);
  }
  return gtanh;
}

float** Matrix::grad_tanh_mat(float **gtanh, float **m, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      gtanh[i][j] = 1.0f - tanhf(m[i][j])*tanhf(m[i][j]);
    }
  }
  return gtanh;
}

float*** Matrix::grad_tanh_mat(float ***gtanh, float ***m, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        gtanh[i][j][k] = 1.0f - tanhf(m[i][j][k])*tanhf(m[i][j][k]);
      }
    }
  }
  return gtanh;
}

float* Matrix::relu_mat(float *relumat, float *m, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    if(m[i] <=0) {
      relumat[i] = 0.0f;
    }
    else {
      relumat[i] = m[i];
    }
  }
  return relumat;
}

float** Matrix::relu_mat(float **relumat, float **m, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      if(m[i][j] <= 0) {
        relumat[i][j] = 0.0f;
      }
      else {
        relumat[i][j] = m[i][j];
      }
    }
  }
  return relumat;
}

float*** Matrix::relu_mat(float ***relumat, float ***m, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        if(m[i][j][k] <= 0) {
          relumat[i][j][k] = 0.0f;
        }
        else {
          relumat[i][j][k] = m[i][j][k];
        }
      }
    }
  }
  return relumat;
}

float* Matrix::grad_relu_mat(float *grelu, float *m, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    if(m[i] <=0) {
      grelu[i] = 0.0f;
    }
    else {
      grelu[i] = 1.0f;
    }
  }
  return grelu;
}

float** Matrix::grad_relu_mat(float **grelu, float **m, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      if(m[i][j] <= 0) {
        grelu[i][j] = 0.0f;
      }
      else {
        grelu[i][j] = 1.0f;
      }
    }
  }
  return grelu;
}

float*** Matrix::grad_relu_mat(float ***grelu, float ***m, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        if(m[i][j][k] <= 0) {
          grelu[i][j][k] = 0.0f;
        }
        else {
          grelu[i][j][k] = 1.0f;
        }
      }
    }
  }
  return grelu;
}

float Matrix::l2norm_mat(float *m, int nx) {
  int i;
  float norm = 0.0f;
  for(i=0; i<nx; i++) {
    norm += m[i] * m[i];
  }
  return norm;
}

float Matrix::l2norm_mat(float **m, int nx, int ny) {
  int i, j;
  float norm = 0.0f;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      norm += m[i][j] * m[i][j];
    }
  }
  return norm;
}

float Matrix::l2norm_mat(float ***m, int nx, int ny, int nz) {
  int i, j, k;
  float norm = 0.0f;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        norm += m[i][j][k] * m[i][j][k];
      }
    }
  }
  return norm;
}

float Matrix::max_mat(float *m, int nx) {
  int i;
  float max_val = 0.0f;

  for(i=0; i<nx; i++) {
    if(m[i] >= max_val)
      max_val = m[i];
  }
  return max_val;
}

float Matrix::max_mat(float **m, int nx, int ny) {
  int i, j;
  float max_val = 0.0f;

  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      if(m[i][j] >= max_val)
        max_val = m[i][j];
    }
  }
  return max_val;
}

float Matrix::max_mat(float ***m, int nx, int ny, int nz) {
  int i, j, k;
  float max_val = 0.0f;

  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        if(m[i][j][k] >= max_val)
          max_val = m[i][j][k];
      }
    }
  }
  return max_val;
}

int Matrix::max_idx_mat(float *m, int nx) {
  int i, idx;
  float max_val = 0.0f;

  for(i=0; i<nx; i++) {
    if(m[i] >= max_val) {
      max_val = m[i];
      idx = i;
    }
  }
  return idx;
}

int* Matrix::max_idx_mat(int *idx, float **m, int nx, int ny) {
  int i, j;
  float max_val = 0.0f;

  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      if(m[i][j] >= max_val) {
        max_val = m[i][j];
        idx[0] = i;
        idx[1] = j;
      }
    }
  }
  return idx;
}

int* Matrix::max_idx_mat(int *idx, float ***m, int nx, int ny, int nz) {
  int i, j, k;
  float max_val = 0.0f;

  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        if(m[i][j][k] >= max_val) {
          max_val = m[i][j][k];
          idx[0] = i;
          idx[1] = j;
          idx[2] = k;
        }
      }
    }
  }
  return idx;
}

void Matrix::print_mat(int *m, int nx) {
  int i;

  for(i=0; i<nx; i++) {
    Serial.printf("%d\t", m[i]);
  }
  Serial.printf("\n");
  return;
}

void Matrix::print_mat(int **m, int nx, int ny) {
  int i, j;

  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      Serial.printf("%d\t", m[i][j]);
    }
    Serial.printf("\n");
  }
  Serial.printf("\n");
  return;
}

void Matrix::print_mat(int ***m, int nx, int ny, int nz) {
  int i, j, k;

  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        Serial.printf("%d\t", m[i][j][k]);
      }
      Serial.printf("\n");
    }
    Serial.printf("\n");
  }
  Serial.printf("\n");
  return;
}

void Matrix::print_mat(float *m, int nx) {
  int i;

  for(i=0; i<nx; i++) {
    Serial.printf("%f\t", m[i]);
  }
  Serial.printf("\n");
  return;
}

void Matrix::print_mat(float **m, int nx, int ny) {
  int i, j;
  
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      Serial.printf("%f\t", m[i][j]);
    }
    Serial.printf("\n");
  }
  Serial.printf("\n");
  return;
}

void Matrix::print_mat(float ***m, int nx, int ny, int nz) {
  int i, j, k;

  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        Serial.printf("%f\t", m[i][j][k]);
      }
      Serial.printf("\n");
    }
    Serial.printf("\n");
  }
  Serial.printf("\n");
  return;
}

float* Matrix::ones_mat(float *ones, int nx) {
  int i;
  for(i=0; i<nx; i++) {
    ones[i] = 1.0f;
  }
  return ones;
}

float** Matrix::ones_mat(float **ones, int nx, int ny) {
  int i, j;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      ones[i][j] = 1.0f;
    }
  }
  return ones;
}

float*** Matrix::ones_mat(float ***ones, int nx, int ny, int nz) {
  int i, j, k;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      for(k=0; k<nz; k++) {
        ones[i][j][k] = 1.0f;
      }
    }
  }
  return ones;
}
