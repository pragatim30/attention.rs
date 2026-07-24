# Proposed Architecture: Pure Rust Feedback-Driven AI/ML Workload Compiler

Based on the SOTA review conducted in #6, here is a proposed high-level system architecture designed natively in Rust to leverage safe concurrency, zero-cost abstractions, and explicit memory layout control.

## System Overview

1. Frontend & Import (PyTorch / ONNX / StableHLO Export)
2. High-Level IR (GraphIR): Transformations & Fusions 
3. Low-Level IR (KernelIR): Tiling & Layout Mapping   
4. Feedback-Driven Auto-Tuner (Profiling Execution)   
5. Backend Codegen (GPU/CUDA via nvptx, CPU, Metal) 

 
## Architectural Breakdown

### 1. Frontend & Ingestion Layer
* **StableHLO / ONNX Importer:** Reads standardized model graphs directly without needing Python runtime dependencies.
* **Rust Native Builder API:** Provides a clean macro-based programmatic API for constructible graphs directly in Rust code.

### 2. High-Level Graph IR (`GraphIR`)
* **Pure Rust SSA Graph Structure:** Uses arenas (`bumpalo` / `petgraph`) for memory-efficient graph manipulation.
* **Optimization Passes:** Operator Fusion (combining point-wise operations with reduction nodes), Dead Code Elimination (DCE), Constant Folding, and Static Shape Propagation.

### 3. Low-Level Kernel IR (`KernelIR`)
* **Tiling & Memory Abstraction:** Converts graph-level tensor ops into micro-blocks (tiles) suitable for GPU shared memory and SIMD registers.
* **Explicit Lifetime Allocation:** Uses Rust’s ownership semantics to compute zero-copy buffer allocations ahead of time, completely eliminating dynamic garbage collection overhead.

### 4. Feedback-Driven Optimization Engine (Auto-Tuner)
* **Measurement Loop:** Executes benchmark iterations of generated candidate kernels on real target hardware.
* **Feedback Collector:** Measures latency, memory throughput, and occupancy metrics.
* **Search Strategy:** Uses evolutionary search / Bayesian optimization to discover optimal block sizes, grid dimensions, and loop unrolling factors dynamically.

### 5. Backend Code Generation
* **LLVM Infrastructure (`inkwell` / native PTX builder):** Generates optimized PTX assembly for NVIDIA GPUs or SPIR-V/Metal code for cross-platform execution.
* **Zero-FFI Cuda/ROCm Executor:** Interoperates with GPU driver APIs directly through safe Rust bindings.
