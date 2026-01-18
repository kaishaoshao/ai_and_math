#ifndef __SK_NN_ARENA_H__
#define __SK_NN_ARENA_H__

#include "sk_nn_config.h"
#include <cstdint>
#include <ks_config.h>

typedef struct  
{
    uint32_t  rows;
    uint32_t  clos;
    sk_f32   *data;   
}sk_matrix;

typedef struct sk_arena{
  struct sk_arena* current; 
  struct sk_arena* prev;
 
  uint64_t  reserve_size;
  uint64_t  commit_size;
   
  
    
}sk_arena;

/// @brief 
/// @param arena 
/// @param rows 
/// @param clos 
/// @return 
sk_matrix* mat_create(sk_arena *arena, uint32_t rows, uint32_t clos);

/// @brief 
/// @param dst 
/// @param src 
void mat_copy(sk_matrix *dst, sk_matrix *src);


void mat_clear(sk_matrix *mat);

void mat_fill(sk_matrix *mat, float val);

void mat_scale(sk_matrix *mat, float val);

sk_bit32 mat_add(sk_matrix *out, const sk_matrix *a, const sk_matrix *b); 

sk_bit32 mat_sub(sk_matrix *out, const sk_matrix *a, const sk_matrix *b);

sk_bit32 mat_mul(sk_matrix *out, const sk_matrix *a, const sk_matrix *b
                 sk_bit8 zer_out, sk_bit8 transpose_a, sk_bit8 transpose_b);


sk_bit32 mat_relu(sk_matrix *out, const sk_matrix *in);

sk_bit32 mat_softmax(sk_matrix *out, const sk_matrix *in); 

sk_bit32 mat_cross_entropy(sk_matrix *out, const sk_matrix *p, const sk_matrix *q);

#endif // __SK_NN_ARENA_H__
