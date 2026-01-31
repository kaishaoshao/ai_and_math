#ifndef __SK_NN_ARENA_H__
#define __SK_NN_ARENA_H__

#include "sk_nn.h"

typedef struct sk_arena{
  struct sk_arena* current;
  struct sk_arena* prev;

  uint64_t  reserve_size;
  uint64_t  commit_size;

} sk_arena;

#define PUSH_STRUCT(sk_arena, T) (T*)arena_push(sk_arena, sizeof(T), FALSE)

#endif // __SK_NN_ARENA_H__
