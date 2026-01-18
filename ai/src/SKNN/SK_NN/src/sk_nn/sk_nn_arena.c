#include "sk_nn_arena.h"


sk_matrix* mat_create(sk_arena *arena, uint32_t rows, uint32_t clos){

}

void mat_copy(sk_matrix *dst, sk_matrix *src){

}


void mat_clear(sk_matrix *mat){

}

void mat_fill(sk_matrix *mat, float val){

}

void mat_scale(sk_matrix *mat, float val){

}

sk_bit32 mat_add(sk_matrix *out, const sk_matrix *a, const sk_matrix *b){

} 

sk_bit32 mat_sub(sk_matrix *out, const sk_matrix *a, const sk_matrix *b){

}

sk_bit32 mat_mul(sk_matrix *out, const sk_matrix *a, const sk_matrix *bl
                 sk_bit8 zer_out, sk_bit8 transpose_a, sk_bit8 transpose_b){

                 }


sk_bit32 mat_relu(sk_matrix *out, const sk_matrix *in){

}

sk_bit32 mat_softmax(sk_matrix *out, const sk_matrix *in){

} 

sk_bit32 mat_cross_entropy(sk_matrix *out, const sk_matrix *p, const sk_matrix *q){

}