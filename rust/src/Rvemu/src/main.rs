

// 定义内存的大小来初始化 CPU
pub const DRAM_SIZE: u64 = 1024 * 1014 * 128


// 64 位的 PC，32 个 64 位的通用整数寄存器以及一个用 u8 向量表示的内存
struct Cpu {
  regs: [u64; 32],
  pc  :  u64,
  dram:  Vec<u8>,
}

// CPU通过将其工作流程划分成多个子过程，以提升其吞吐量和性能。
// 经典的 MIPS 流水线架构包含以下五个部分
// 1.取指：根据pc值读取内存中的指令 (fetch)
// 2.解码：解码指令 (decode)
// 3.执行：执行指令 (execute)
// 4.访存：将结果写回内存 (memory access)
// 5.写回：将运算结果（包括新的PC值）写回寄存器 (write back)


// 因为栈指针 (SP) 需要指向栈顶（内存的最高地址），
// 同时，我们将 PC 置 0，意味着我们的程序将从内存地址 0 处开始执行
impl Cpu {

// 构造函数：创建新的 CPU 实例
// DRAM 布局：
// ┌─────────────────────┐ 0x0000_0000
// │                     │
// │    程序代码 (code)   │ ← 从地址 0 开始加载
// │                     │
// ├─────────────────────┤
// │                     │
// │    未使用空间         │
// │                     │
// ├─────────────────────┤ DRAM_SIZE (0x0800_0000 = 128MB)
// │    栈向下增长         │ ← regs[2] = sp 指向这里
// │    ↓↓↓↓↓            │
// └─────────────────────┘

// 寄存器：
// regs[0]  = 0      (x0/zero, 硬连线为 0)
// regs[2]  = DRAM_SIZE  (x2/sp, 栈指针)
// ...
// regs[31] = 0      (x31/t6)
// pc = 0            从第一条指令执行
  fn new(code : Vec<u8>) -> Self {
    // 初始化寄存器数组（32 个，全置 0）
    let mut regs = [0; 32];
    // 设置栈指针（x2/sp）指向 DRAM 顶部
    regs[2] = DRAM_SIZE;
    // 返回构造的 Self（Cpu 实例
    Self {
      regs,       // 寄存器组
      pc: 0,      // 程序计数器从0开始
      dram: code  // 代码加载到内存
    }
  }

//   DRAM 内存（Vec<u8>）：
// 地址（u32）    索引（usize）    数据
// 0x0000_0000 → 0              [0x93]
// 0x0000_0001 → 1              [0x01]
// 0x0000_0002 → 2              [0x00]
// 0x0000_0003 → 3              [0x00]
// ...
// self.pc = 0x0000_0000 (u32)
//     ↓ as usize
// index = 0 (usize)
//     ↓
// self.dram[0] = 0x93
// 位 31-24   位 23-16   位 15-8    位 7-0
// [  0x00  ][  0x00  ][  0x01  ][  0x93  ]
//     ↑          ↑         ↑         ↑
//   byte3      byte2     byte1     byte0
//   (左移24)   (左移16)   (左移8)   (不移动)
// 由于 RISC-V 指令是 32 位的，因此，内存需要读取的是 [pc, pc+1, pc+2, pc+3]
// 这四个地址上的值，并组合成一个 32 位的指令
    fn fetch(&mut self) -> u32 {
    // as usize
    let index = self.pc as usize;
    let inst = self.dram[index] as u32
            | ((self.dram[index + 1] as u32) << 8)
            | ((self.dram[index + 2] as u32) << 16)
            | ((self.dram[index + 3] as u32) << 24)
  }


}




fn main() {
  println!("Hello, world!");
}

