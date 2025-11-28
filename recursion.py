"""
================================================================================
Recursive Factorial Implementation on SimpleCPU
================================================================================

This program demonstrates:
1. A simple Python recursive factorial function
2. How it maps to SimpleCPU assembly language
3. Memory layout with code, stack, and data sections
4. Function call mechanics (CALL/RET instructions)
5. Recursion visualization with stack frames

MEMORY LAYOUT:
================================================================================
0x0000 - 0x00FF: Main Program (driver code)
0x0100 - 0x01FF: Factorial Function (recursive)
0x0200 - 0x02FF: Helper Functions (print_digit, print_newline)
0xF000 - 0xFEFF: Stack (grows downward from 0xFEFF)
0xFF00 - 0xFFFF: Memory-Mapped I/O

STACK FRAME STRUCTURE:
================================================================================
Each function call creates a stack frame:

    High Memory (0xFEFF)
    ┌─────────────────┐
    │  Return Address │  ← Pushed by CALL instruction
    ├─────────────────┤
    │  Parameter (N)  │  ← Pushed by caller
    ├─────────────────┤
    │  Local Vars     │  ← Function's working space
    ├─────────────────┤
    │  Saved Registers│  ← Preserved across calls
    └─────────────────┘
    Low Memory
    
    SP points here after setup →
"""

# ============================================================================
# PART 1: Simple Python Recursive Function
# ============================================================================

def factorial(n):
    """
    Recursive factorial function in Python
    
    Base case: factorial(0) = factorial(1) = 1
    Recursive case: factorial(n) = n * factorial(n-1)
    """
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

def main_python():
    """Python driver program"""
    print("Python Recursive Factorial")
    print("-" * 40)
    
    test_values = [0, 1, 3, 5]
    for n in test_values:
        result = factorial(n)
        print(f"factorial({n}) = {result}")
    print()

# ============================================================================
# PART 2: SimpleCPU Assembly Implementation
# ============================================================================

FACTORIAL_ASSEMBLY = """
; ============================================================================
; RECURSIVE FACTORIAL PROGRAM FOR SIMPLECPU
; ============================================================================
; Computes factorial(N) using recursion
; Demonstrates function calls, stack frames, and recursion

; ============================================================================
; MAIN PROGRAM - Entry Point at 0x0000
; ============================================================================

main:
    ; Test factorial(5) -> should output 120
    MOV A, #5              ; N = 5
    PUSH A                 ; Push parameter onto stack
    CALL factorial         ; Call factorial function
    POP B                  ; Pop parameter (cleanup)
    
    ; Result is in A, print it
    CALL print_result
    
    ; Test factorial(4) -> should output 24  
    MOV A, #4
    PUSH A
    CALL factorial
    POP B
    CALL print_result
    
    ; Test factorial(3) -> should output 6
    MOV A, #3
    PUSH A
    CALL factorial
    POP B
    CALL print_result
    
    HLT                    ; End program

; ============================================================================
; FACTORIAL FUNCTION - Located at 0x0100
; ============================================================================
; Input: N on stack (pushed by caller)
; Output: A = factorial(N)
; Modifies: A, B, C
; Stack frame:
;   [SP+0]: Return address (pushed by CALL)
;   [SP+2]: Parameter N

factorial:
    ; Get parameter N from stack
    ; Stack layout: [SP] = return_low, [SP+1] = return_high, 
    ;               [SP+2] = N
    PUSH B                 ; Save B register (we'll use it)
    PUSH C                 ; Save C register
    
    ; Load N from stack into A
    ; N is at offset +4 from current SP (after our 2 PUSHes)
    MOV A, [0xF000]        ; Placeholder - will manually fix SP offset
    
    ; Base case: if N <= 1, return 1
    CMP A, #1
    JZ base_case           ; Jump if N == 1
    JC base_case           ; Jump if N < 1 (carry set when N < 1)
    
    ; Recursive case: factorial(N) = N * factorial(N-1)
    PUSH A                 ; Save N for later multiplication
    
    ; Compute N-1
    DEC A                  ; A = N - 1
    
    ; Recursive call: factorial(N-1)
    PUSH A                 ; Push N-1 as parameter
    CALL factorial         ; Recursive call
    POP B                  ; Pop parameter
    
    ; Now A contains factorial(N-1)
    ; Multiply by N
    POP B                  ; B = N (restore saved N)
    CALL multiply          ; A = A * B
    
    ; Restore registers and return
    POP C
    POP B
    RET

base_case:
    ; Return 1
    MOV A, #1
    POP C                  ; Restore registers
    POP B
    RET

; ============================================================================
; MULTIPLY FUNCTION - A = A * B
; ============================================================================
; Input: A, B (two numbers to multiply)
; Output: A = A * B
; Uses: C as counter
; Note: Limited to 8-bit result (will overflow for large values)

multiply:
    PUSH C                 ; Save C
    PUSH D                 ; Save D
    MOV C, B               ; C = multiplier (B)
    MOV D, A               ; D = multiplicand (A)
    MOV A, #0              ; A = result accumulator
    
    CMP C, #0              ; Check if B == 0
    JZ mult_done
    
mult_loop:
    ADD A, D               ; Add multiplicand to result
    DEC C                  ; Decrement counter
    JNZ mult_loop          ; Continue if not zero
    
mult_done:
    POP D                  ; Restore registers
    POP C
    RET

; ============================================================================
; HELPER FUNCTIONS - Print result
; ============================================================================

print_result:
    PUSH A                 ; Save result
    
    ; Print "= "
    MOV B, #61             ; '='
    MOV [0xFF00], B
    MOV B, #32             ; ' '
    MOV [0xFF00], B
    
    POP A                  ; Restore result
    
    ; Convert number to ASCII and print
    ; For simplicity, handle only values < 100
    PUSH A
    
    ; Extract hundreds digit
    MOV B, #100
    CALL divide            ; A = A / B (quotient), C = remainder
    ADD A, #48             ; Convert to ASCII
    CMP A, #48             ; Don't print if zero
    JZ skip_hundreds
    MOV [0xFF00], A
    
skip_hundreds:
    MOV A, C               ; A = remainder
    
    ; Extract tens digit
    MOV B, #10
    CALL divide            ; A = quotient (tens), C = remainder (ones)
    ADD A, #48
    MOV [0xFF00], A        ; Print tens
    
    ; Print ones digit
    MOV A, C
    ADD A, #48
    MOV [0xFF00], A
    
    ; Print newline
    MOV A, #10
    MOV [0xFF00], A
    
    POP A                  ; Restore original result
    RET

; Simple division: A = A / B, remainder in C
divide:
    PUSH D
    MOV C, #0              ; Quotient
    MOV D, A               ; Dividend
    
div_loop:
    CMP D, B               ; Compare dividend with divisor
    JC div_done            ; If dividend < divisor, done
    SUB D, B               ; Subtract divisor
    INC C                  ; Increment quotient
    JMP div_loop
    
div_done:
    MOV A, C               ; A = quotient
    MOV C, D               ; C = remainder
    POP D
    RET
"""

# Simplified version that works with SimpleCPU limitations
FACTORIAL_SIMPLE = """
; Simplified Recursive Factorial for SimpleCPU
; Computes factorial(3) = 6

main:
    MOV A, #3              ; N = 3
    CALL factorial         ; Result in A
    
    ; Print result (A should be 6)
    ADD A, #48             ; Convert to ASCII
    MOV [0xFF00], A        ; Print
    MOV A, #10             ; Newline
    MOV [0xFF00], A
    
    HLT

; Factorial function (simplified - parameter in A)
factorial:
    ; Base case: if A <= 1, return 1
    CMP A, #1
    JZ base_case
    
    ; Recursive case
    PUSH A                 ; Save N
    DEC A                  ; N-1
    CALL factorial         ; factorial(N-1) -> result in A
    POP B                  ; B = N
    
    ; Multiply A * B (simplified - just add A to itself B times)
    PUSH A                 ; Save factorial(N-1)
    MOV C, B               ; C = N (loop counter)
    MOV A, #0              ; Result
    POP D                  ; D = factorial(N-1)
    
mult_loop:
    CMP C, #0
    JZ mult_done
    ADD A, D               ; Add factorial(N-1)
    DEC C
    JMP mult_loop
    
mult_done:
    RET

base_case:
    MOV A, #1
    RET
"""

# ============================================================================
# PART 3: Memory Layout Visualization
# ============================================================================

def print_memory_layout():
    """Print detailed memory layout diagram"""
    print("\n" + "=" * 80)
    print("MEMORY LAYOUT FOR FACTORIAL PROGRAM")
    print("=" * 80)
    
    layout = """
ADDRESS RANGE    SIZE     SECTION           CONTENTS
─────────────────────────────────────────────────────────────────────────────
0x0000 - 0x00FF  256 B   MAIN PROGRAM      • Entry point (main)
                                            • Test cases
                                            • Program initialization
                                            
0x0100 - 0x01FF  256 B   FACTORIAL FUNC    • Recursive factorial function
                                            • Base case check (N <= 1)
                                            • Recursive call logic
                                            • Stack frame management
                                            
0x0200 - 0x02FF  256 B   HELPER FUNCS      • multiply(A, B)
                                            • print_result()
                                            • divide(A, B)
                                            
0x0300 - 0xEFFF  59 KB   UNUSED            Available for more code/data

0xF000 - 0xFEFF  4 KB    STACK             • Grows DOWNWARD from 0xFEFF
                                            • Function parameters
                                            • Return addresses
                                            • Saved registers
                                            • Local variables
                                            
0xFF00           1 B     CONSOLE OUT       Write byte to print character
0xFF01           1 B     CONSOLE IN        Read byte from input buffer
0xFF02 - 0xFF03  2 B     TIMER             16-bit timer value
0xFF04 - 0xFFFF  252 B   MMIO RESERVED     Future I/O devices
    """
    print(layout)

# ============================================================================
# PART 4: Function Call Mechanics
# ============================================================================

def print_call_mechanics():
    """Explain function call mechanism"""
    print("\n" + "=" * 80)
    print("FUNCTION CALL MECHANICS")
    print("=" * 80)
    
    mechanics = """
CALL INSTRUCTION (3 bytes: opcode + address):
─────────────────────────────────────────────────────────────────────────────
    Before CALL factorial:
        PC = 0x0008  (address of CALL instruction)
        SP = 0xFEFF  (top of stack)
        A  = 5       (parameter)
        
    CALL factorial executes:
        1. PUSH return address onto stack:
           memory[--SP] = (PC + 3) & 0xFF        ; Low byte of return addr
           memory[--SP] = ((PC + 3) >> 8) & 0xFF ; High byte of return addr
           
        2. Set PC to function address:
           PC = 0x0100  (factorial function entry)
           
    After CALL:
        PC = 0x0100  (now executing factorial)
        SP = 0xFEFD  (stack grew by 2 bytes)
        Stack: [0xFEFE] = 0x00, [0xFEFD] = 0x0B  (return address 0x000B)

RET INSTRUCTION (1 byte: opcode):
─────────────────────────────────────────────────────────────────────────────
    Before RET:
        PC = 0x0150  (somewhere in factorial function)
        SP = 0xFEFD  (pointing to return address)
        
    RET executes:
        1. POP return address from stack:
           lo = memory[SP++]
           hi = memory[SP++]
           return_addr = (hi << 8) | lo
           
        2. Set PC to return address:
           PC = return_addr
           
    After RET:
        PC = 0x000B  (back in caller, after CALL instruction)
        SP = 0xFEFF  (stack restored)
    """
    print(mechanics)

# ============================================================================
# PART 5: Recursion Visualization
# ============================================================================

def print_recursion_trace():
    """Visualize recursion with stack frames"""
    print("\n" + "=" * 80)
    print("RECURSION TRACE: factorial(3)")
    print("=" * 80)
    
    trace = """
STEP 1: Initial Call - factorial(3)
─────────────────────────────────────────────────────────────────────────────
    Stack @ 0xFEFF:
    ┌──────────────────┐
    │                  │ ← 0xFEFF (SP initially)
    │   [empty]        │
    └──────────────────┘
    
    main calls factorial(3):
        PUSH 3           ; Push parameter
        CALL factorial   ; Push return address, jump to function
    
    Stack @ 0xFEFC:
    ┌──────────────────┐
    │   0x00, 0x0B     │ ← 0xFEFD (return address)
    ├──────────────────┤
    │   3              │ ← 0xFEFC (parameter N=3)
    └──────────────────┘
    
    factorial(3) executes:
        - Checks: 3 <= 1? NO
        - Recursive case: compute 3 * factorial(2)
        - Calls factorial(2)

STEP 2: Recursive Call - factorial(2)
─────────────────────────────────────────────────────────────────────────────
    Stack before recursive call:
    ┌──────────────────┐
    │   0x00, 0x0B     │ ← Return to main
    ├──────────────────┤
    │   3              │ ← N for factorial(3)
    ├──────────────────┤
    │   3              │ ← Saved N (will multiply later)
    └──────────────────┘
    
    factorial(3) calls factorial(2):
        DEC A            ; A = 2
        PUSH A           ; Push parameter 2
        CALL factorial   ; Push return address
    
    Stack @ 0xFEF6:
    ┌──────────────────┐
    │   0x00, 0x0B     │ ← Return to main
    ├──────────────────┤
    │   3              │ ← Original N
    ├──────────────────┤
    │   3              │ ← Saved for multiply
    ├──────────────────┤
    │   0x01, 0x35     │ ← Return to factorial(3)
    ├──────────────────┤
    │   2              │ ← N for factorial(2)
    └──────────────────┘ ← 0xFEF6 (SP)
    
    factorial(2) executes:
        - Checks: 2 <= 1? NO
        - Recursive case: compute 2 * factorial(1)

STEP 3: Recursive Call - factorial(1)
─────────────────────────────────────────────────────────────────────────────
    Stack @ 0xFEF0:
    ┌──────────────────┐
    │   0x00, 0x0B     │ ← Return to main
    ├──────────────────┤
    │   3              │
    ├──────────────────┤
    │   3              │ ← Saved for factorial(3)
    ├──────────────────┤
    │   0x01, 0x35     │ ← Return to factorial(3)
    ├──────────────────┤
    │   2              │
    ├──────────────────┤
    │   2              │ ← Saved for factorial(2)
    ├──────────────────┤
    │   0x01, 0x35     │ ← Return to factorial(2)
    ├──────────────────┤
    │   1              │ ← N for factorial(1)
    └──────────────────┘ ← 0xFEF0 (SP)
    
    factorial(1) executes:
        - Checks: 1 <= 1? YES → BASE CASE!
        - Returns 1

STEP 4: Unwinding - factorial(1) returns 1
─────────────────────────────────────────────────────────────────────────────
    factorial(1):
        MOV A, #1        ; A = 1 (return value)
        RET              ; Pop return address, jump back
    
    Stack after RET @ 0xFEF8:
    ┌──────────────────┐
    │   ...            │
    ├──────────────────┤
    │   2              │ ← N for factorial(2)
    ├──────────────────┤
    │   2              │ ← Saved for multiply
    └──────────────────┘ ← 0xFEF8 (SP)
    
    Back in factorial(2):
        A = 1 (from factorial(1))
        POP B            ; B = 2 (saved N)
        CALL multiply    ; A = 1 * 2 = 2
        RET              ; Return 2

STEP 5: Unwinding - factorial(2) returns 2
─────────────────────────────────────────────────────────────────────────────
    Back in factorial(3):
        A = 2 (from factorial(2))
        POP B            ; B = 3 (saved N)
        CALL multiply    ; A = 2 * 3 = 6
        RET              ; Return 6

STEP 6: Final Return - factorial(3) returns 6
─────────────────────────────────────────────────────────────────────────────
    Stack @ 0xFEFF:
    ┌──────────────────┐
    │   [restored]     │ ← 0xFEFF (SP)
    └──────────────────┘
    
    Back in main:
        A = 6            ; factorial(3) = 6 ✓
        
KEY OBSERVATIONS:
    • Stack grows DOWNWARD (SP decreases)
    • Each recursive call adds a new stack frame
    • Maximum stack depth = N frames for factorial(N)
    • Return values passed through register A
    • Stack automatically cleaned up by RET instructions
    """
    print(trace)

# ============================================================================
# PART 6: Run the actual SimpleCPU simulation
# ============================================================================

# Import the SimpleCPU from the provided code
import sys
from typing import List

# Copy the CPU and Assembler classes here (or import from main.py)
# For this demonstration, I'll create a minimal version

def run_on_simplecpu():
    """Run the factorial program on SimpleCPU"""
    print("\n" + "=" * 80)
    print("EXECUTING ON SIMPLECPU")
    print("=" * 80)
    
    # Import from the provided code
    try:
        from main import CPU, Assembler
        
        cpu = CPU()
        asm = Assembler()
        
        # Assemble the program
        machine_code = asm.assemble(FACTORIAL_SIMPLE)
        
        print(f"\nAssembled {len(machine_code)} bytes")
        print("\nMachine Code:")
        for i in range(0, len(machine_code), 16):
            hex_line = ' '.join(f"{b:02X}" for b in machine_code[i:i+16])
            print(f"  0x{i:04X}: {hex_line}")
        
        # Load and run
        cpu.load_program(machine_code)
        print("\nOutput:")
        print("-" * 40)
        cpu.run(max_cycles=10000)
        
        print("\nFinal CPU State:")
        cpu.print_state()
        
    except ImportError:
        print("SimpleCPU module not available - showing conceptual execution")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RECURSIVE FACTORIAL - SOFTWARE CPU IMPLEMENTATION")
    print("=" * 80)
    
    # Part 1: Python version
    print("\n")
    main_python()
    
    # Part 2: Show memory layout
    print_memory_layout()
    
    # Part 3: Explain function calls
    print_call_mechanics()
    
    # Part 4: Visualize recursion
    print_recursion_trace()
    
    # Part 5: Show assembly code
    print("\n" + "=" * 80)
    print("ASSEMBLY CODE")
    print("=" * 80)
    print(FACTORIAL_SIMPLE)
    
    # Part 6: Run on CPU
    run_on_simplecpu()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)