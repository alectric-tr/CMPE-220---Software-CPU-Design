"""
================================================================================
SimpleCPU - A Complete Software CPU Implementation
================================================================================

CPU ARCHITECTURE SCHEMATIC:
    
    ┌──────────────────────────────────────────────────────────────┐
    │                         MEMORY (64KB)                        │
    │  0x0000-0xEFFF: Program Memory                              │
    │  0xF000-0xFEFF: Stack                                       │
    │  0xFF00-0xFFFF: Memory Mapped I/O                          │
    └──────────────────┬───────────────────────────────────────────┘
                       │ BUS (16-bit address, 8-bit data)
    ┌──────────────────┴───────────────────────────────────────────┐
    │                      CONTROL UNIT                            │
    │  • Instruction Fetch    • Decode    • Execute               │
    │  • Program Counter (PC) • Instruction Register (IR)         │
    └──────────────────┬───────────────────────────────────────────┘
                       │
    ┌──────────────────┴───────────────────────────────────────────┐
    │                        REGISTERS                             │
    │  A, B, C, D (8-bit general purpose)                        │
    │  SP (16-bit stack pointer)  FLAGS (Z, C, N, O)             │
    └──────────────────┬───────────────────────────────────────────┘
                       │
    ┌──────────────────┴───────────────────────────────────────────┐
    │                           ALU                                │
    │  • Arithmetic Ops  • Logic Ops  • Comparisons               │
    └──────────────────────────────────────────────────────────────┘

ISA SPECIFICATION:
=================

INSTRUCTION FORMAT:
  - 1-3 bytes per instruction
  - Byte 0: Opcode (8 bits)
  - Byte 1: Operand 1 (optional, 8 bits)
  - Byte 2: Operand 2 (optional, 8 bits)

ADDRESSING MODES:
  - Immediate:  MOV A, #42    (value is in instruction)
  - Register:   MOV A, B      (register to register)
  - Direct:     MOV A, [0x100] (memory address)
  - Indirect:   MOV A, [B]    (address in register)

FLAGS:
  - Z (Zero):     Set if result is 0
  - C (Carry):    Set if carry/borrow occurred
  - N (Negative): Set if result is negative (bit 7 set)
  - O (Overflow): Set if signed overflow occurred

MEMORY MAP:
  0x0000-0xEFFF: Program Memory (60KB)
  0xF000-0xFEFF: Stack (4KB)
  0xFF00:        Console Output (write byte to print)
  0xFF01:        Console Input (read byte from buffer)
  0xFF02:        Timer LSB
  0xFF03:        Timer MSB
"""

import sys
from typing import List, Dict, Optional, Tuple

# ============================================================================
# INSTRUCTION SET ENCODING
# ============================================================================

class Opcode:
    """Instruction opcodes for SimpleCPU"""
    # Data Movement
    MOV_REG_IMM   = 0x01  # MOV A, #42
    MOV_REG_REG   = 0x02  # MOV A, B
    MOV_REG_MEM   = 0x03  # MOV A, [0x100]
    MOV_MEM_REG   = 0x04  # MOV [0x100], A
    MOV_REG_IND   = 0x05  # MOV A, [B]
    
    # Arithmetic
    ADD_REG_IMM   = 0x10  # ADD A, #5
    ADD_REG_REG   = 0x11  # ADD A, B
    SUB_REG_IMM   = 0x12  # SUB A, #5
    SUB_REG_REG   = 0x13  # SUB A, B
    INC_REG       = 0x14  # INC A
    DEC_REG       = 0x15  # DEC A
    
    # Logic
    AND_REG_REG   = 0x20  # AND A, B
    OR_REG_REG    = 0x21  # OR A, B
    XOR_REG_REG   = 0x22  # XOR A, B
    NOT_REG       = 0x23  # NOT A
    
    # Shifts
    SHL_REG       = 0x30  # SHL A (shift left)
    SHR_REG       = 0x31  # SHR A (shift right)
    
    # Comparison
    CMP_REG_IMM   = 0x40  # CMP A, #42
    CMP_REG_REG   = 0x41  # CMP A, B
    
    # Jumps
    JMP           = 0x50  # JMP 0x100
    JZ            = 0x51  # JZ 0x100 (jump if zero)
    JNZ           = 0x52  # JNZ 0x100 (jump if not zero)
    JC            = 0x53  # JC 0x100 (jump if carry)
    JNC           = 0x54  # JNC 0x100 (jump if not carry)
    
    # Stack
    PUSH_REG      = 0x60  # PUSH A
    POP_REG       = 0x61  # POP A
    CALL          = 0x62  # CALL 0x100
    RET           = 0x63  # RET
    
    # Special
    NOP           = 0x00  # No operation
    HLT           = 0xFF  # Halt CPU

# Register encoding
REG_A = 0
REG_B = 1
REG_C = 2
REG_D = 3

REGISTER_NAMES = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

# ============================================================================
# CPU EMULATOR
# ============================================================================

class CPU:
    """SimpleCPU Emulator with full instruction set"""
    
    def __init__(self):
        # Registers (8-bit general purpose)
        self.reg = [0] * 4  # A, B, C, D
        
        # Special registers
        self.pc = 0x0000      # Program Counter (16-bit)
        self.sp = 0xF000      # Stack Pointer (starts at stack base)
        self.ir = 0x00        # Instruction Register
        
        # Flags
        self.flag_z = False   # Zero
        self.flag_c = False   # Carry
        self.flag_n = False   # Negative
        self.flag_o = False   # Overflow
        
        # Memory (64KB)
        self.memory = [0] * 0x10000
        
        # Memory-mapped I/O
        self.console_output = []
        self.console_input = []
        self.timer = 0
        
        # Control
        self.halted = False
        self.cycle_count = 0
        
        # Debugging
        self.trace = False
        
    # ------------------------------------------------------------------------
    # MEMORY ACCESS
    # ------------------------------------------------------------------------
    
    def read_byte(self, address: int) -> int:
        """Read a byte from memory with MMIO support"""
        address &= 0xFFFF
        
        # Memory-mapped I/O
        if address == 0xFF00:  # Console output (write-only, reads as 0)
            return 0
        elif address == 0xFF01:  # Console input
            if self.console_input:
                return self.console_input.pop(0)
            return 0
        elif address == 0xFF02:  # Timer LSB
            return self.timer & 0xFF
        elif address == 0xFF03:  # Timer MSB
            return (self.timer >> 8) & 0xFF
        
        return self.memory[address]
    
    def write_byte(self, address: int, value: int):
        """Write a byte to memory with MMIO support"""
        address &= 0xFFFF
        value &= 0xFF
        
        # Memory-mapped I/O
        if address == 0xFF00:  # Console output
            self.console_output.append(value)
            if 32 <= value <= 126 or value in (10, 13):  # Printable or newline
                print(chr(value), end='')
            return
        elif address == 0xFF01:  # Console input (read-only)
            return
        elif address == 0xFF02:  # Timer LSB (read-only)
            return
        elif address == 0xFF03:  # Timer MSB (read-only)
            return
        
        self.memory[address] = value
    
    def read_word(self, address: int) -> int:
        """Read a 16-bit word (little-endian)"""
        lo = self.read_byte(address)
        hi = self.read_byte(address + 1)
        return (hi << 8) | lo
    
    def write_word(self, address: int, value: int):
        """Write a 16-bit word (little-endian)"""
        value &= 0xFFFF
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)
    
    # ------------------------------------------------------------------------
    # ALU OPERATIONS
    # ------------------------------------------------------------------------
    
    def update_flags(self, result: int, check_carry: bool = False, 
                     carry: bool = False, check_overflow: bool = False,
                     a: int = 0, b: int = 0):
        """Update CPU flags based on ALU operation result"""
        result &= 0xFF
        
        self.flag_z = (result == 0)
        self.flag_n = (result & 0x80) != 0
        
        if check_carry:
            self.flag_c = carry
        
        if check_overflow:
            # Overflow if sign of operands same but result different
            sign_a = (a & 0x80) != 0
            sign_b = (b & 0x80) != 0
            sign_r = (result & 0x80) != 0
            self.flag_o = (sign_a == sign_b) and (sign_a != sign_r)
    
    def alu_add(self, a: int, b: int) -> int:
        """Add two 8-bit values"""
        result = a + b
        carry = result > 0xFF
        self.update_flags(result, check_carry=True, carry=carry, 
                         check_overflow=True, a=a, b=b)
        return result & 0xFF
    
    def alu_sub(self, a: int, b: int) -> int:
        """Subtract two 8-bit values"""
        result = a - b
        borrow = result < 0
        self.update_flags(result, check_carry=True, carry=borrow,
                         check_overflow=True, a=a, b=b)
        return result & 0xFF
    
    def alu_and(self, a: int, b: int) -> int:
        """Bitwise AND"""
        result = a & b
        self.update_flags(result)
        return result
    
    def alu_or(self, a: int, b: int) -> int:
        """Bitwise OR"""
        result = a | b
        self.update_flags(result)
        return result
    
    def alu_xor(self, a: int, b: int) -> int:
        """Bitwise XOR"""
        result = a ^ b
        self.update_flags(result)
        return result
    
    def alu_not(self, a: int) -> int:
        """Bitwise NOT"""
        result = (~a) & 0xFF
        self.update_flags(result)
        return result
    
    def alu_shl(self, a: int) -> int:
        """Shift left"""
        result = a << 1
        carry = (result & 0x100) != 0
        self.update_flags(result, check_carry=True, carry=carry)
        return result & 0xFF
    
    def alu_shr(self, a: int) -> int:
        """Shift right"""
        carry = (a & 0x01) != 0
        result = a >> 1
        self.update_flags(result, check_carry=True, carry=carry)
        return result
    
    # ------------------------------------------------------------------------
    # STACK OPERATIONS
    # ------------------------------------------------------------------------
    
    def push_byte(self, value: int):
        """Push a byte onto the stack"""
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, value)
    
    def pop_byte(self) -> int:
        """Pop a byte from the stack"""
        value = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return value
    
    def push_word(self, value: int):
        """Push a 16-bit word onto the stack"""
        self.push_byte((value >> 8) & 0xFF)  # High byte
        self.push_byte(value & 0xFF)         # Low byte
    
    def pop_word(self) -> int:
        """Pop a 16-bit word from the stack"""
        lo = self.pop_byte()
        hi = self.pop_byte()
        return (hi << 8) | lo
    
    # ------------------------------------------------------------------------
    # INSTRUCTION EXECUTION (Fetch-Decode-Execute Cycle)
    # ------------------------------------------------------------------------
    
    def fetch(self) -> int:
        """FETCH: Read instruction from memory at PC"""
        opcode = self.read_byte(self.pc)
        self.ir = opcode
        self.pc = (self.pc + 1) & 0xFFFF
        return opcode
    
    def fetch_operand(self) -> int:
        """Fetch an operand byte"""
        operand = self.read_byte(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        return operand
    
    def fetch_address(self) -> int:
        """Fetch a 16-bit address"""
        addr = self.read_word(self.pc)
        self.pc = (self.pc + 2) & 0xFFFF
        return addr
    
    def execute(self, opcode: int):
        """EXECUTE: Execute the decoded instruction"""
        
        # Data Movement Instructions
        if opcode == Opcode.MOV_REG_IMM:
            reg = self.fetch_operand()
            value = self.fetch_operand()
            self.reg[reg] = value
            
        elif opcode == Opcode.MOV_REG_REG:
            dest = self.fetch_operand()
            src = self.fetch_operand()
            self.reg[dest] = self.reg[src]
            
        elif opcode == Opcode.MOV_REG_MEM:
            reg = self.fetch_operand()
            addr = self.fetch_address()
            self.reg[reg] = self.read_byte(addr)
            
        elif opcode == Opcode.MOV_MEM_REG:
            addr = self.fetch_address()
            reg = self.fetch_operand()
            self.write_byte(addr, self.reg[reg])
            
        elif opcode == Opcode.MOV_REG_IND:
            dest = self.fetch_operand()
            src = self.fetch_operand()
            addr = self.reg[src]
            self.reg[dest] = self.read_byte(addr)
        
        # Arithmetic Instructions
        elif opcode == Opcode.ADD_REG_IMM:
            reg = self.fetch_operand()
            value = self.fetch_operand()
            self.reg[reg] = self.alu_add(self.reg[reg], value)
            
        elif opcode == Opcode.ADD_REG_REG:
            dest = self.fetch_operand()
            src = self.fetch_operand()
            self.reg[dest] = self.alu_add(self.reg[dest], self.reg[src])
            
        elif opcode == Opcode.SUB_REG_IMM:
            reg = self.fetch_operand()
            value = self.fetch_operand()
            self.reg[reg] = self.alu_sub(self.reg[reg], value)
            
        elif opcode == Opcode.SUB_REG_REG:
            dest = self.fetch_operand()
            src = self.fetch_operand()
            self.reg[dest] = self.alu_sub(self.reg[dest], self.reg[src])
            
        elif opcode == Opcode.INC_REG:
            reg = self.fetch_operand()
            self.reg[reg] = self.alu_add(self.reg[reg], 1)
            
        elif opcode == Opcode.DEC_REG:
            reg = self.fetch_operand()
            self.reg[reg] = self.alu_sub(self.reg[reg], 1)
        
        # Logic Instructions
        elif opcode == Opcode.AND_REG_REG:
            dest = self.fetch_operand()
            src = self.fetch_operand()
            self.reg[dest] = self.alu_and(self.reg[dest], self.reg[src])
            
        elif opcode == Opcode.OR_REG_REG:
            dest = self.fetch_operand()
            src = self.fetch_operand()
            self.reg[dest] = self.alu_or(self.reg[dest], self.reg[src])
            
        elif opcode == Opcode.XOR_REG_REG:
            dest = self.fetch_operand()
            src = self.fetch_operand()
            self.reg[dest] = self.alu_xor(self.reg[dest], self.reg[src])
            
        elif opcode == Opcode.NOT_REG:
            reg = self.fetch_operand()
            self.reg[reg] = self.alu_not(self.reg[reg])
        
        # Shift Instructions
        elif opcode == Opcode.SHL_REG:
            reg = self.fetch_operand()
            self.reg[reg] = self.alu_shl(self.reg[reg])
            
        elif opcode == Opcode.SHR_REG:
            reg = self.fetch_operand()
            self.reg[reg] = self.alu_shr(self.reg[reg])
        
        # Comparison Instructions
        elif opcode == Opcode.CMP_REG_IMM:
            reg = self.fetch_operand()
            value = self.fetch_operand()
            self.alu_sub(self.reg[reg], value)  # Sets flags but doesn't store
            
        elif opcode == Opcode.CMP_REG_REG:
            reg1 = self.fetch_operand()
            reg2 = self.fetch_operand()
            self.alu_sub(self.reg[reg1], self.reg[reg2])
        
        # Jump Instructions
        elif opcode == Opcode.JMP:
            addr = self.fetch_address()
            self.pc = addr
            
        elif opcode == Opcode.JZ:
            addr = self.fetch_address()
            if self.flag_z:
                self.pc = addr
                
        elif opcode == Opcode.JNZ:
            addr = self.fetch_address()
            if not self.flag_z:
                self.pc = addr
                
        elif opcode == Opcode.JC:
            addr = self.fetch_address()
            if self.flag_c:
                self.pc = addr
                
        elif opcode == Opcode.JNC:
            addr = self.fetch_address()
            if not self.flag_c:
                self.pc = addr
        
        # Stack Instructions
        elif opcode == Opcode.PUSH_REG:
            reg = self.fetch_operand()
            self.push_byte(self.reg[reg])
            
        elif opcode == Opcode.POP_REG:
            reg = self.fetch_operand()
            self.reg[reg] = self.pop_byte()
            
        elif opcode == Opcode.CALL:
            addr = self.fetch_address()
            self.push_word(self.pc)
            self.pc = addr
            
        elif opcode == Opcode.RET:
            self.pc = self.pop_word()
        
        # Special Instructions
        elif opcode == Opcode.NOP:
            pass  # Do nothing
            
        elif opcode == Opcode.HLT:
            self.halted = True
            
        else:
            raise ValueError(f"Unknown opcode: 0x{opcode:02X}")
    
    def step(self):
        """Execute one instruction cycle (Fetch-Execute)"""
        if self.halted:
            return
        
        # Update timer
        self.timer = (self.timer + 1) & 0xFFFF
        
        # Trace execution
        if self.trace:
            self.print_state()
        
        # FETCH phase
        opcode = self.fetch()
        
        # EXECUTE phase
        self.execute(opcode)
        
        self.cycle_count += 1
    
    def run(self, max_cycles: int = 100000):
        """Run the CPU until halted or max cycles reached"""
        while not self.halted and self.cycle_count < max_cycles:
            self.step()
        
        if self.cycle_count >= max_cycles:
            print(f"\n[CPU] Execution stopped: max cycles ({max_cycles}) reached")
        else:
            print(f"\n[CPU] Halted after {self.cycle_count} cycles")
    
    # ------------------------------------------------------------------------
    # DEBUGGING AND UTILITIES
    # ------------------------------------------------------------------------
    
    def load_program(self, program: List[int], start_address: int = 0x0000):
        """Load a program into memory"""
        for i, byte in enumerate(program):
            self.memory[start_address + i] = byte & 0xFF
        self.pc = start_address
        print(f"[CPU] Loaded {len(program)} bytes at 0x{start_address:04X}")
    
    def print_state(self):
        """Print current CPU state"""
        flags = f"{'Z' if self.flag_z else '-'}{'C' if self.flag_c else '-'}" \
                f"{'N' if self.flag_n else '-'}{'O' if self.flag_o else '-'}"
        print(f"PC:0x{self.pc:04X} IR:0x{self.ir:02X} " \
              f"A:{self.reg[0]:3d} B:{self.reg[1]:3d} " \
              f"C:{self.reg[2]:3d} D:{self.reg[3]:3d} " \
              f"SP:0x{self.sp:04X} F:{flags}")
    
    def dump_memory(self, start: int, length: int):
        """Dump memory contents"""
        print(f"\nMemory Dump (0x{start:04X} - 0x{start+length-1:04X}):")
        for i in range(0, length, 16):
            addr = start + i
            hex_data = ' '.join(f"{self.memory[addr+j]:02X}" 
                               for j in range(min(16, length-i)))
            ascii_data = ''.join(chr(self.memory[addr+j]) 
                                if 32 <= self.memory[addr+j] <= 126 
                                else '.' 
                                for j in range(min(16, length-i)))
            print(f"0x{addr:04X}: {hex_data:<48} {ascii_data}")

# ============================================================================
# ASSEMBLER
# ============================================================================

class Assembler:
    """SimpleCPU Assembler - converts assembly language to machine code"""
    
    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.machine_code: List[int] = []
        self.current_address = 0
        
        # Instruction encoding table
        self.instructions = {
            'MOV': self._assemble_mov,
            'ADD': self._assemble_add,
            'SUB': self._assemble_sub,
            'INC': self._assemble_inc,
            'DEC': self._assemble_dec,
            'AND': self._assemble_and,
            'OR': self._assemble_or,
            'XOR': self._assemble_xor,
            'NOT': self._assemble_not,
            'SHL': self._assemble_shl,
            'SHR': self._assemble_shr,
            'CMP': self._assemble_cmp,
            'JMP': self._assemble_jmp,
            'JZ': self._assemble_jz,
            'JNZ': self._assemble_jnz,
            'JC': self._assemble_jc,
            'JNC': self._assemble_jnc,
            'PUSH': self._assemble_push,
            'POP': self._assemble_pop,
            'CALL': self._assemble_call,
            'RET': self._assemble_ret,
            'NOP': self._assemble_nop,
            'HLT': self._assemble_hlt,
        }
    
    def assemble(self, source_code: str) -> List[int]:
        """Assemble source code into machine code"""
        # First pass: collect labels
        lines = self._preprocess(source_code)
        self._first_pass(lines)
        
        # Second pass: generate machine code
        self.current_address = 0
        self.machine_code = []
        self._second_pass(lines)
        
        return self.machine_code
    
    def _preprocess(self, source_code: str) -> List[str]:
        """Remove comments and empty lines"""
        lines = []
        for line in source_code.split('\n'):
            # Remove comments
            if ';' in line:
                line = line[:line.index(';')]
            line = line.strip()
            if line:
                lines.append(line)
        return lines
    
    def _first_pass(self, lines: List[str]):
        """First pass: collect labels and calculate addresses"""
        address = 0
        for line in lines:
            if ':' in line:
                label = line[:line.index(':')].strip()
                self.labels[label] = address
                line = line[line.index(':')+1:].strip()
                if not line:
                    continue
            
            # Estimate instruction length
            parts = line.split()
            if parts:
                mnemonic = parts[0].upper()
                length = self._estimate_instruction_length(mnemonic, parts[1:] if len(parts) > 1 else [])
                address += length
    
    def _second_pass(self, lines: List[str]):
        """Second pass: generate machine code"""
        for line in lines:
            # Skip labels
            if ':' in line:
                line = line[line.index(':')+1:].strip()
                if not line:
                    continue
            
            parts = line.split(',')
            parts = [p.strip() for part in parts for p in part.split()]
            
            if not parts:
                continue
            
            mnemonic = parts[0].upper()
            operands = [p for p in parts[1:] if p != ',']
            
            if mnemonic in self.instructions:
                self.instructions[mnemonic](operands)
            else:
                raise ValueError(f"Unknown instruction: {mnemonic}")
    
    def _estimate_instruction_length(self, mnemonic: str, operands: List[str]) -> int:
        """Estimate instruction length in bytes"""
        lengths = {
            'NOP': 1, 'HLT': 1, 'RET': 1,
            'INC': 2, 'DEC': 2, 'NOT': 2, 'SHL': 2, 'SHR': 2,
            'PUSH': 2, 'POP': 2,
            'JMP': 3, 'JZ': 3, 'JNZ': 3, 'JC': 3, 'JNC': 3, 'CALL': 3,
        }
        
        if mnemonic in lengths:
            return lengths[mnemonic]
        
        # Most other instructions are 3 bytes
        return 3
    
    def _emit(self, *bytes_to_emit):
        """Emit bytes to machine code"""
        for byte in bytes_to_emit:
            self.machine_code.append(byte & 0xFF)
            self.current_address += 1
    
    def _parse_register(self, operand: str) -> int:
        """Parse register name to register number"""
        operand = operand.upper().strip()
        if operand == 'A':
            return REG_A
        elif operand == 'B':
            return REG_B
        elif operand == 'C':
            return REG_C
        elif operand == 'D':
            return REG_D
        else:
            raise ValueError(f"Invalid register: {operand}")
    
    def _parse_value(self, operand: str) -> int:
        """Parse numeric literal or label"""
        operand = operand.strip()
        
        # Check if it's a label
        if operand in self.labels:
            return self.labels[operand]
        
        # Remove # prefix if present
        if operand.startswith('#'):
            operand = operand[1:]
        
        # Parse hex or decimal
        if operand.startswith('0x') or operand.startswith('0X'):
            return int(operand, 16)
        else:
            return int(operand)
    
    def _parse_address(self, operand: str) -> int:
        """Parse memory address [0x100] or label"""
        operand = operand.strip()
        
        # Remove brackets if present
        if operand.startswith('[') and operand.endswith(']'):
            operand = operand[1:-1].strip()
        
        return self._parse_value(operand)
    
    # Instruction assemblers
    def _assemble_mov(self, operands: List[str]):
        dest = operands[0]
        src = operands[1]
        
        # MOV reg, #imm
        if src.startswith('#'):
            reg = self._parse_register(dest)
            value = self._parse_value(src)
            self._emit(Opcode.MOV_REG_IMM, reg, value)
        
        # MOV reg, [addr]
        elif src.startswith('['):
            reg = self._parse_register(dest)
            # Check if indirect [B] or direct [0x100]
            inner = src[1:-1].strip()
            if inner.upper() in ['A', 'B', 'C', 'D']:
                # Indirect addressing
                src_reg = self._parse_register(inner)
                self._emit(Opcode.MOV_REG_IND, reg, src_reg)
            else:
                # Direct addressing
                addr = self._parse_address(src)
                self._emit(Opcode.MOV_REG_MEM, reg, addr & 0xFF, (addr >> 8) & 0xFF)
        
        # MOV [addr], reg
        elif dest.startswith('['):
            addr = self._parse_address(dest)
            reg = self._parse_register(src)
            self._emit(Opcode.MOV_MEM_REG, addr & 0xFF, (addr >> 8) & 0xFF, reg)
        
        # MOV reg, reg
        else:
            dest_reg = self._parse_register(dest)
            src_reg = self._parse_register(src)
            self._emit(Opcode.MOV_REG_REG, dest_reg, src_reg)
    
    def _assemble_add(self, operands: List[str]):
        dest = self._parse_register(operands[0])
        src = operands[1]
        if src.startswith('#'):
            value = self._parse_value(src)
            self._emit(Opcode.ADD_REG_IMM, dest, value)
        else:
            src_reg = self._parse_register(src)
            self._emit(Opcode.ADD_REG_REG, dest, src_reg)
    
    def _assemble_sub(self, operands: List[str]):
        dest = self._parse_register(operands[0])
        src = operands[1]
        if src.startswith('#'):
            value = self._parse_value(src)
            self._emit(Opcode.SUB_REG_IMM, dest, value)
        else:
            src_reg = self._parse_register(src)
            self._emit(Opcode.SUB_REG_REG, dest, src_reg)
    
    def _assemble_inc(self, operands: List[str]):
        reg = self._parse_register(operands[0])
        self._emit(Opcode.INC_REG, reg)
    
    def _assemble_dec(self, operands: List[str]):
        reg = self._parse_register(operands[0])
        self._emit(Opcode.DEC_REG, reg)
    
    def _assemble_and(self, operands: List[str]):
        dest = self._parse_register(operands[0])
        src = self._parse_register(operands[1])
        self._emit(Opcode.AND_REG_REG, dest, src)
    
    def _assemble_or(self, operands: List[str]):
        dest = self._parse_register(operands[0])
        src = self._parse_register(operands[1])
        self._emit(Opcode.OR_REG_REG, dest, src)
    
    def _assemble_xor(self, operands: List[str]):
        dest = self._parse_register(operands[0])
        src = self._parse_register(operands[1])
        self._emit(Opcode.XOR_REG_REG, dest, src)
    
    def _assemble_not(self, operands: List[str]):
        reg = self._parse_register(operands[0])
        self._emit(Opcode.NOT_REG, reg)
    
    def _assemble_shl(self, operands: List[str]):
        reg = self._parse_register(operands[0])
        self._emit(Opcode.SHL_REG, reg)
    
    def _assemble_shr(self, operands: List[str]):
        reg = self._parse_register(operands[0])
        self._emit(Opcode.SHR_REG, reg)
    
    def _assemble_cmp(self, operands: List[str]):
        dest = self._parse_register(operands[0])
        src = operands[1]
        if src.startswith('#'):
            value = self._parse_value(src)
            self._emit(Opcode.CMP_REG_IMM, dest, value)
        else:
            src_reg = self._parse_register(src)
            self._emit(Opcode.CMP_REG_REG, dest, src_reg)
    
    def _assemble_jmp(self, operands: List[str]):
        addr = self._parse_value(operands[0])
        self._emit(Opcode.JMP, addr & 0xFF, (addr >> 8) & 0xFF)
    
    def _assemble_jz(self, operands: List[str]):
        addr = self._parse_value(operands[0])
        self._emit(Opcode.JZ, addr & 0xFF, (addr >> 8) & 0xFF)
    
    def _assemble_jnz(self, operands: List[str]):
        addr = self._parse_value(operands[0])
        self._emit(Opcode.JNZ, addr & 0xFF, (addr >> 8) & 0xFF)
    
    def _assemble_jc(self, operands: List[str]):
        addr = self._parse_value(operands[0])
        self._emit(Opcode.JC, addr & 0xFF, (addr >> 8) & 0xFF)
    
    def _assemble_jnc(self, operands: List[str]):
        addr = self._parse_value(operands[0])
        self._emit(Opcode.JNC, addr & 0xFF, (addr >> 8) & 0xFF)
    
    def _assemble_push(self, operands: List[str]):
        reg = self._parse_register(operands[0])
        self._emit(Opcode.PUSH_REG, reg)
    
    def _assemble_pop(self, operands: List[str]):
        reg = self._parse_register(operands[0])
        self._emit(Opcode.POP_REG, reg)
    
    def _assemble_call(self, operands: List[str]):
        addr = self._parse_value(operands[0])
        self._emit(Opcode.CALL, addr & 0xFF, (addr >> 8) & 0xFF)
    
    def _assemble_ret(self, operands: List[str]):
        self._emit(Opcode.RET)
    
    def _assemble_nop(self, operands: List[str]):
        self._emit(Opcode.NOP)
    
    def _assemble_hlt(self, operands: List[str]):
        self._emit(Opcode.HLT)

# ============================================================================
# EXAMPLE PROGRAMS
# ============================================================================

# Example 1: Timer Program (demonstrates Fetch/Compute/Store cycles)
TIMER_PROGRAM = """
; Timer Example - Demonstrates CPU execution cycles
; This program reads the timer and displays it

start:
    MOV A, #0          ; Initialize counter
    
loop:
    MOV B, [0xFF02]    ; FETCH: Read timer LSB from MMIO
    MOV C, [0xFF03]    ; FETCH: Read timer MSB from MMIO
    
    ; COMPUTE: Add timer value to accumulator
    ADD A, B           ; Add LSB
    
    ; STORE: Write result to console
    MOV [0xFF00], A    ; Write to console output
    
    ; Check if we've done 10 iterations
    INC D              ; Increment loop counter
    CMP D, #10         ; COMPUTE: Compare with 10
    JZ done            ; Jump if zero flag set
    
    JMP loop           ; Continue loop
    
done:
    HLT                ; Halt CPU
"""

# Example 2: Hello World
HELLO_WORLD_PROGRAM = """
; Hello, World! Program
; Prints "Hello, World!" to console

start:
    MOV A, #0          ; Initialize string pointer
    
print_loop:
    MOV B, A           ; Copy pointer to B
    ADD B, #data       ; Add offset to data
    MOV C, [B]         ; Load character (indirect addressing simulation)
    
    ; Check for null terminator
    CMP C, #0
    JZ done
    
    ; Print character
    MOV [0xFF00], C
    
    ; Next character
    INC A
    JMP print_loop
    
done:
    HLT

; Data section (we'll manually place this)
data:
"""

# Better Hello World using direct addressing
HELLO_WORLD_SIMPLE = """
; Simple Hello, World!
; Prints "Hello!" to console output

start:
    MOV A, #72         ; 'H'
    MOV [0xFF00], A
    
    MOV A, #101        ; 'e'
    MOV [0xFF00], A
    
    MOV A, #108        ; 'l'
    MOV [0xFF00], A
    MOV [0xFF00], A    ; second 'l'
    
    MOV A, #111        ; 'o'
    MOV [0xFF00], A
    
    MOV A, #33         ; '!'
    MOV [0xFF00], A
    
    MOV A, #10         ; newline
    MOV [0xFF00], A
    
    HLT
"""

# Example 3: Fibonacci Sequence
FIBONACCI_PROGRAM = """
; Fibonacci Sequence Generator
; Computes first N Fibonacci numbers
; Results stored in memory starting at 0x0200

start:
    MOV A, #0          ; fib(0) = 0
    MOV B, #1          ; fib(1) = 1
    MOV C, #10         ; Count (compute 10 numbers)
    MOV D, #0          ; Index counter
    
    ; Store first two numbers
    MOV [0x0200], A    ; Store fib(0)
    MOV [0x0201], B    ; Store fib(1)
    
    MOV D, #2          ; Start from index 2
    
fib_loop:
    ; Check if done
    CMP D, C
    JZ print_results
    
    ; Compute next Fibonacci: A = A + B
    PUSH A             ; Save old A
    ADD A, B           ; A = A + B (new fib number)
    
    ; Shift values: B = old A, A = new value
    POP B              ; B = old A
    PUSH A             ; Save new A
    
    ; Store result in memory
    ; Address = 0x0200 + D
    PUSH B
    MOV B, #0x00
    ADD B, D
    ; Simulate storing at 0x0200 + D
    ; (Limited indirect addressing)
    
    POP B
    POP A
    
    ; For simplicity, just print to console
    PUSH A
    MOV A, B
    ADD A, #48         ; Convert to ASCII digit (works for 0-9)
    MOV [0xFF00], A
    MOV A, #32         ; Space
    MOV [0xFF00], A
    POP A
    
    INC D
    JMP fib_loop
    
print_results:
    MOV A, #10         ; Newline
    MOV [0xFF00], A
    HLT
"""

FIBONACCI_SIMPLE = """
; Fibonacci - prints first 7 single-digit numbers (0,1,1,2,3,5,8)

start:
    MOV A, #0          ; fib(n-2)
    MOV B, #1          ; fib(n-1)
    MOV C, #5          ; counter (5 more after printing 0 and 1)

    ; Print 0
    MOV D, #48
    MOV [0xFF00], D
    MOV D, #32
    MOV [0xFF00], D
    
    ; Print 1
    MOV D, #49
    MOV [0xFF00], D
    MOV D, #32
    MOV [0xFF00], D

loop:
    ; Check if done first
    CMP C, #0
    JZ done
    DEC C
    
    ; Compute next: new_val = A + B
    PUSH A             ; Save old A
    ADD A, B           ; A now = A + B (new fibonacci number)
    
    ; Print the new number
    PUSH A             ; Save the new fib value
    ADD A, #48         ; Convert to ASCII
    MOV [0xFF00], A    ; Print it
    MOV A, #32         ; Space
    MOV [0xFF00], A
    POP A              ; Restore the fib value
    
    ; Shift values: new A = old B, new B = new fib value
    POP D              ; D = old A
    MOV D, B           ; D = old B (this will become new A)
    MOV B, A           ; B = new fib value
    MOV A, D           ; A = old B
    
    JMP loop

done:
    MOV A, #10
    MOV [0xFF00], A
    HLT
"""

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main program - demonstrates the CPU with example programs"""
    
    print("=" * 70)
    print("SimpleCPU - Complete Software CPU Implementation")
    print("=" * 70)
    print()
    
    # Create CPU and Assembler
    cpu = CPU()
    asm = Assembler()
    
    # Menu
    programs = {
        '1': ('Timer Example', TIMER_PROGRAM),
        '2': ('Hello World', HELLO_WORLD_SIMPLE),
        '3': ('Fibonacci Sequence', FIBONACCI_SIMPLE),
    }
    
    print("Available Programs:")
    for key, (name, _) in programs.items():
        print(f"  {key}. {name}")
    print()
    
    choice = input("Select program (1-3) or Enter for all: ").strip()
    
    if not choice:
        # Run all programs
        choices = ['1', '2', '3']
    else:
        choices = [choice]
    
    for choice in choices:
        if choice not in programs:
            print(f"Invalid choice: {choice}")
            continue
        
        name, source = programs[choice]
        
        print("\n" + "=" * 70)
        print(f"Running: {name}")
        print("=" * 70)
        
        # Show source code
        print("\nSource Code:")
        print("-" * 70)
        print(source)
        print("-" * 70)
        
        # Assemble
        try:
            machine_code = asm.assemble(source)
            print(f"\nAssembled: {len(machine_code)} bytes")
            
            # Show machine code
            print("\nMachine Code (hex):")
            for i in range(0, len(machine_code), 16):
                hex_line = ' '.join(f"{b:02X}" for b in machine_code[i:i+16])
                print(f"  0x{i:04X}: {hex_line}")
            
            # Reset CPU and load program
            cpu = CPU()
            cpu.load_program(machine_code)
            
            # Show execution for timer example
            if choice == '1':
                print("\n" + "-" * 70)
                print("Execution Trace (Fetch-Decode-Execute Cycles):")
                print("-" * 70)
                cpu.trace = True
                cpu.run(max_cycles=100)
            else:
                print("\nOutput:")
                print("-" * 70)
                cpu.run(max_cycles=10000)
            
            # Show final state
            print("\n" + "-" * 70)
            print("Final CPU State:")
            print("-" * 70)
            cpu.print_state()
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SimpleCPU execution complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
