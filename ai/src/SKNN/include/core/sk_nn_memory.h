#ifndef __SK_NN_ARENA_H__
#define __SK_NN_ARENA_H__

#include "sk_nn.h"

typedef struct sk_memory{
  struct sk_memory* current;
  struct sk_memory* prev;

  uint64_t  reserve_size;
  uint64_t  commit_size;

} sk_memory;

#define PUSH_STRUCT(sk_memory, T) (T*)arena_push(sk_memory, sizeof(T), FALSE)


#endif // __SK_NN_ARENA_H__
