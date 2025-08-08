use crate::kernel::linux::{LinuxArchConfig, Syscall};

pub fn config() -> LinuxArchConfig {
    LinuxArchConfig {
        syscall_num_register: "w8".to_owned(),
        arg_registers: [
            "x0".to_owned(),
            "x1".to_owned(),
            "x2".to_owned(),
            "x3".to_owned(),
            "x4".to_owned(),
            "x5".to_owned(),
        ],
        return_register: "x0".to_owned(),
        // https://github.com/torvalds/linux/blob/v6.16/arch/arm64/tools/syscall_64.tbl
        // https://github.com/torvalds/linux/blob/v6.16/scripts/syscall.tbl
        syscall_map: [
            (96, Syscall::SetTidAddress),
            (73, Syscall::Ppoll),
            (134, Syscall::RtSigAction),
            (132, Syscall::SigAltStack),
            (222, Syscall::Mmap),
            (226, Syscall::Mprotect),
            (135, Syscall::RtSigProcMask),
            (214, Syscall::Brk),
            (64, Syscall::Write),
            (215, Syscall::Munmap),
            (94, Syscall::ExitGroup),
        ]
        .into_iter()
        .collect(),
    }
}
