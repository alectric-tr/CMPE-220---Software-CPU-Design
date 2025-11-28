# Software CPU and Recursive Factorial Example

This project implements a complete software-based CPU in Python along with an assembler and example programs. One of the examples demonstrates a recursive factorial function written both in Python and in the custom assembly language supported by this CPU.

## Overview

The software CPU includes:

- A 64 KB memory model
- Four 8-bit general-purpose registers
- A 16-bit program counter and stack pointer
- Arithmetic, logic, comparison, shift, and jump instructions
- CALL and RET support for function calls and recursion
- A memory-mapped console output device

The assembler converts the custom assembly language into machine code, which can then be executed by the CPU emulator.

## Files

- `main.py`  
  Contains the full CPU implementation, assembler, and several example programs.

- `recursion.py`  
  Shows a recursive factorial function in both Python and the custom assembly language, explains memory layout and call mechanics, and runs the assembled code on the CPU.

## Running

Run the recursion demonstration:

```
python recursion.py
```

Run the CPU examples menu:

```
python main.py
```

## Purpose

This project demonstrates how a CPU can be modeled entirely in software and how higher-level concepts such as recursion map onto low-level execution using a custom instruction set and stack-based function calls.
