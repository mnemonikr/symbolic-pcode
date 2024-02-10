use std::borrow::Cow;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use sla::{
    Address, AddressSpace, AddressSpaceType, BoolOp, IntOp, IntSign, OpCode, PcodeInstruction,
    VarnodeData,
};
use sym::SymbolicByte;
use symbolic_pcode::emulator::{PcodeEmulator, StandardPcodeEmulator};
use symbolic_pcode::mem::{Memory, SymbolicMemory};

const fn processor_space() -> AddressSpace {
    AddressSpace {
        id: 1,
        name: Cow::Borrowed("processor_space"),
        word_size: 1,
        address_size: 8,
        space_type: AddressSpaceType::Processor,
        big_endian: false,
    }
}

const fn constant_space() -> AddressSpace {
    AddressSpace {
        id: 2,
        name: Cow::Borrowed("constant_space"),
        word_size: 1,
        address_size: 8,
        space_type: AddressSpaceType::Constant,
        big_endian: false,
    }
}

const fn internal_space() -> AddressSpace {
    AddressSpace {
        id: 0,
        name: Cow::Borrowed("internal_space"),
        word_size: 1,
        address_size: 8,
        space_type: AddressSpaceType::Internal,
        big_endian: false,
    }
}

fn setup_copy() -> (Memory, PcodeInstruction) {
    let data = 0x1122334455667788u64
        .to_le_bytes()
        .into_iter()
        .map(SymbolicByte::from);
    let mut memory = Memory::new();

    let pcode_copy = PcodeInstruction {
        address: Address {
            offset: 0xffff,
            address_space: processor_space(),
        },
        op_code: OpCode::Copy,
        inputs: vec![VarnodeData {
            address: Address {
                offset: 0x1234,
                address_space: processor_space(),
            },
            size: 8,
        }],
        output: Some(VarnodeData {
            address: Address {
                offset: 0x5678,
                address_space: processor_space(),
            },
            size: 8,
        }),
    };

    memory
        .write(&pcode_copy.inputs[0], data)
        .expect("failed to initialize memory");

    (memory, pcode_copy)
}

fn setup_load() -> (Memory, PcodeInstruction) {
    const PROCESSOR_SPACE: AddressSpace = processor_space();
    const CONSTANT_SPACE: AddressSpace = constant_space();
    const INTERNAL_SPACE: AddressSpace = internal_space();
    let pcode_load = PcodeInstruction {
        address: Address {
            offset: 0xffff,
            address_space: PROCESSOR_SPACE,
        },
        op_code: OpCode::Load,
        inputs: vec![
            VarnodeData {
                address: Address {
                    offset: PROCESSOR_SPACE.id as u64,
                    address_space: CONSTANT_SPACE,
                },
                size: 8,
            },
            VarnodeData {
                address: Address {
                    offset: 0x1234,
                    address_space: PROCESSOR_SPACE,
                },
                size: 8,
            },
        ],
        output: Some(VarnodeData {
            address: Address {
                offset: 0,
                address_space: INTERNAL_SPACE,
            },
            size: 8,
        }),
    };

    let mut memory = Memory::new();

    // Write indirect offset to memory
    let offset = 0x5678u64;
    let indirect_offset = offset.to_le_bytes().into_iter().map(SymbolicByte::from);
    memory
        .write(&pcode_load.inputs[1], indirect_offset)
        .expect("failed to write indirect offset");

    // Write data to memory
    let data = 0x1122334455667788u64
        .to_le_bytes()
        .into_iter()
        .map(SymbolicByte::from);
    let data_varnode = VarnodeData {
        address: Address {
            offset,
            address_space: PROCESSOR_SPACE,
        },
        size: 8,
    };
    memory
        .write(&data_varnode, data)
        .expect("failed to write data");

    (memory, pcode_load)
}

fn setup_store() -> (Memory, PcodeInstruction) {
    const PROCESSOR_SPACE: AddressSpace = processor_space();
    const CONSTANT_SPACE: AddressSpace = constant_space();
    let instruction = PcodeInstruction {
        address: Address {
            offset: 0xffff,
            address_space: PROCESSOR_SPACE,
        },
        op_code: OpCode::Store,
        inputs: vec![
            VarnodeData {
                address: Address {
                    offset: PROCESSOR_SPACE.id as u64,
                    address_space: CONSTANT_SPACE,
                },
                size: 8,
            },
            VarnodeData {
                address: Address {
                    offset: 0x1234,
                    address_space: PROCESSOR_SPACE,
                },
                size: 8,
            },
            VarnodeData {
                address: Address {
                    offset: 0x4321,
                    address_space: PROCESSOR_SPACE,
                },
                size: 8,
            },
        ],
        output: None,
    };

    let mut memory = Memory::new();

    // Write indirect offset to memory
    let offset = 0x5678u64;
    let indirect_offset = offset.to_le_bytes().into_iter().map(SymbolicByte::from);
    memory
        .write(&instruction.inputs[1], indirect_offset)
        .expect("failed to write indirect offset");

    // Write data to memory
    let data = 0x1122334455667788u64
        .to_le_bytes()
        .into_iter()
        .map(SymbolicByte::from);
    memory
        .write(&instruction.inputs[2], data)
        .expect("failed to write data");

    (memory, instruction)
}

fn create_arithmetic_setup_fn(op_code: OpCode) -> impl FnMut() -> (Memory, PcodeInstruction) {
    move || {
        let input_size = match op_code {
            OpCode::Bool(_) => 1,
            _ => 8,
        };

        let inputs = match op_code {
            OpCode::Bool(BoolOp::Negate) | OpCode::Int(IntOp::Bitwise(BoolOp::Negate)) => {
                vec![VarnodeData {
                    address: Address {
                        address_space: internal_space(),
                        offset: 0,
                    },
                    size: input_size,
                }]
            }
            _ => vec![
                VarnodeData {
                    address: Address {
                        address_space: internal_space(),
                        offset: 0,
                    },
                    size: input_size,
                },
                VarnodeData {
                    address: Address {
                        address_space: internal_space(),
                        offset: 8,
                    },
                    size: input_size,
                },
            ],
        };

        let output_size = match op_code {
            OpCode::Int(IntOp::Carry(_) | IntOp::Borrow) => 1,
            OpCode::Int(
                IntOp::Equal | IntOp::NotEqual | IntOp::LessThan(_) | IntOp::LessThanOrEqual(_),
            ) => 1,
            OpCode::Bool(_) => 1,
            _ => 8,
        };

        let mut memory = Memory::new();
        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_space(),
                offset: 0xFF00000000,
            },
            op_code,
            inputs,
            output: Some(VarnodeData {
                address: Address {
                    address_space: internal_space(),
                    offset: 16,
                },
                size: output_size,
            }),
        };

        memory
            .write(
                &instruction.inputs[0],
                0xfedcba9876543210u64
                    .to_le_bytes()
                    .into_iter()
                    .map(SymbolicByte::from)
                    .take(instruction.inputs[0].size),
            )
            .expect("failed to write lhs");

        if instruction.inputs.len() > 1 {
            memory
                .write(
                    &instruction.inputs[1],
                    0x0123456789abcdefu64
                        .to_le_bytes()
                        .into_iter()
                        .map(SymbolicByte::from)
                        .take(instruction.inputs[1].size),
                )
                .expect("failed to write rhs");
        }

        (memory, instruction)
    }
}

pub fn standard_emulator(c: &mut Criterion) {
    let emulator = StandardPcodeEmulator::new([internal_space(), processor_space()]);

    let mut bench_arithmetic = |op_code| {
        let mut group = c.benchmark_group("Emulate");
        group.bench_function(
            BenchmarkId::new("Arithmetic", format!("{op_code:?}")),
            |b| {
                b.iter_batched(
                    create_arithmetic_setup_fn(op_code),
                    |mut data| {
                        emulator
                            .emulate(&mut data.0, &data.1)
                            .expect("failed to emulate instruction")
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.finish();
    };

    bench_arithmetic(OpCode::Int(IntOp::Subtract));
    bench_arithmetic(OpCode::Int(IntOp::Borrow));
    bench_arithmetic(OpCode::Int(IntOp::Multiply));
    bench_arithmetic(OpCode::Int(IntOp::Divide(IntSign::Unsigned)));
    bench_arithmetic(OpCode::Int(IntOp::Divide(IntSign::Signed)));
    bench_arithmetic(OpCode::Int(IntOp::Remainder(IntSign::Unsigned)));
    bench_arithmetic(OpCode::Int(IntOp::Remainder(IntSign::Signed)));
    bench_arithmetic(OpCode::Int(IntOp::Add));
    bench_arithmetic(OpCode::Int(IntOp::Carry(IntSign::Unsigned)));
    bench_arithmetic(OpCode::Int(IntOp::Carry(IntSign::Signed)));
    bench_arithmetic(OpCode::Int(IntOp::Equal));
    bench_arithmetic(OpCode::Int(IntOp::NotEqual));
    bench_arithmetic(OpCode::Int(IntOp::LessThan(IntSign::Unsigned)));
    bench_arithmetic(OpCode::Int(IntOp::LessThan(IntSign::Signed)));
    bench_arithmetic(OpCode::Int(IntOp::LessThanOrEqual(IntSign::Unsigned)));
    bench_arithmetic(OpCode::Int(IntOp::LessThanOrEqual(IntSign::Signed)));
    bench_arithmetic(OpCode::Int(IntOp::Bitwise(BoolOp::And)));
    bench_arithmetic(OpCode::Int(IntOp::Bitwise(BoolOp::Or)));
    bench_arithmetic(OpCode::Int(IntOp::Bitwise(BoolOp::Xor)));
    bench_arithmetic(OpCode::Int(IntOp::Bitwise(BoolOp::Negate)));
    bench_arithmetic(OpCode::Bool(BoolOp::And));
    bench_arithmetic(OpCode::Bool(BoolOp::Or));
    bench_arithmetic(OpCode::Bool(BoolOp::Xor));
    bench_arithmetic(OpCode::Bool(BoolOp::Negate));

    c.bench_function("copy", |b| {
        b.iter_batched(
            setup_copy,
            |mut data| {
                emulator
                    .emulate(&mut data.0, &data.1)
                    .expect("failed to emulate instruction")
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("load", |b| {
        b.iter_batched(
            setup_load,
            |mut data| {
                emulator
                    .emulate(&mut data.0, &data.1)
                    .expect("failed to emulate instruction")
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("store", |b| {
        b.iter_batched(
            setup_store,
            |mut data| {
                emulator
                    .emulate(&mut data.0, &data.1)
                    .expect("failed to emulate instruction")
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, standard_emulator);
criterion_main!(benches);
