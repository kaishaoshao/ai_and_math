#ifndef __SK_NN_ARENA_H__
#define __SK_NN_ARENA_H__

// [---------------reserve_size-------------------]
// [--------commit_size-----][------free_size-----]
// ^start                   ^commit_pos           ^end
// [allocater][unused]
//            ^pos (pos < commit_pos)           
typedef struct sk_mem{
  // Total size of virtual address space reserved（maximu capacity）
  sk_u64  reserve_size;   
  // Granularity/step size for committing physical memmory
  sk_u64  commit_size;
  // Current allocation offset 
  sk_u64  pos;
  // Offset of the boundary of currently committed memory
  sk_u64  commit_pos;
} sk_mem;

typedef struct sk_tmp_memory{
  // Pointer to the parent memory 
  sk_mem mem;
  // The allocation position at the snapshot was created(save point)
  sk_u64 start_pos;

}sk_tmp_mem;

sk_mem* mem_create(sk_u64 reserve_size, sk_u64 commit_size);

void mem_destroy(sk_mem* mem);

void* mem_push(sk_mem* mem, sk_u64 size, sk_bool non_zero);

void mem_pop(sk_mem* mem, sk_u64 size);

void mem_pop_to(sk_mem* mem, sk_u64 pos);

void mem_clear(sk_mem* mem);

sk_mem_tmp mem_temp_begin(sk_mem* mem);

void mem_temp_end(sk_mem_tmp temp);

sk_mem_tmp mem_scratch_get(sk_mem* conflicts, sk_u32 num_conflicts);

void mem_scratch_release(sk_mem_temp scratch);



#define PUSH_STRUCT(sk_mem, T) (T*)mem_push(sk_mem, sizeof(T), FALSE)


#endif // __SK_NN_ARENA_H__
