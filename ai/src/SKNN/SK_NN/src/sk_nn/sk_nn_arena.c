#include "sk_nn/sk_nn_arena.h"


sk_matrix* mat_create(sk_arena *arena, uint32_t rows, uint32_t cols){
  sk_matrix *mat = PUSH_STRUCT(arena, sk_matrix);

  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, sk_f32, (uint64_t)rows * cols);
  return mat;
}

bool mat_copy(sk_matrix *dst, sk_matrix *src){
  if (dsk->rows != src->rows || dsk->cols != src->cols)
    return false;
  memcpy(dst->data, src->data, sizeof(sk_f32) * (uint64_t)dst->rows * dst->cols);
  return true;
}


void mat_clear(sk_matrix *mat){
  memset(mat->data, 0, sizeof(sk_f32) * (uint64_t)mat->rows * mat.cols);
}

void mat_fill(sk_matrix *mat, float val){
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  for (uint64_t i = 0; i < size; i++)
    mat->data[i] = val;
}

void mat_scale(sk_matrix *mat, float scale) {
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  for (uint64_t i = 0; i < size; i++)
  {
    mat->data[i] *= scale;
  }
}

void mat_sum(sk_matrix *mat) {
  uint64_t size = (uint64_t)mat->rows * mat->cols;
  sk_f32 sum = 0.0f;
  for (uint64_t i = 0; i < size; i++)
    sum += mat->data[i];
  return sum;
}

sk_b32 mat_add(sk_matrix *out, const sk_matrix *a, const sk_matrix *b){
  if (a->rows != b->rows || a->cols != b->cols)
    return false;
  if (out->rows != a->rows || out->cols != a->cols)
    return false;
  uint64_t size = (uint64_t)out->rows * out->cols;
  for (uint64_t i = 0; i < size; i++)
    out->data[i] = a->data[i] + b->data[i];
  return false;
}

sk_b32 mat_sub(sk_matrix *out, const sk_matrix *a, const sk_matrix *b){
  if (a->rows != b->rows || a->cols != b->cols)
    return false;
  if (out->rows != a->rows || out->cols != a->cols)
    return false;
  uint64_t size = (uint64_t)out->rows * out->cols;
  for (uint64_t i = 0; i < size; i++)
    out->data[i] = a->data[i] + b->data[i];
  return false;
}

sk_b32 mat_mul(sk_matrix *out, const sk_matrix *a, const sk_matrix *b
               sk_b8 zer_out, sk_b8 transpose_a, sk_b8 transpose_b){

}


sk_b32 mat_relu(sk_matrix *out, const sk_matrix *in){

}

sk_b32 mat_softmax(sk_matrix *out, const sk_matrix *in){

}

sk_b32 mat_cross_entropy(sk_matrix *out, const sk_matrix *p, const sk_matrix *q){

}