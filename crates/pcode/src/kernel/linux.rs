use std::collections::BTreeSet;
use std::ops::Range;

use log::{error, trace, warn};

use crate::emulator::{self, ControlFlow};
use crate::kernel::Kernel;
use crate::mem::VarnodeDataStore;
use libsla::{Address, Sleigh, VarnodeData};
use pcode_ops::convert::{PcodeValue, TryFromPcodeValueError};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Memory(#[from] crate::mem::Error),

    #[error(transparent)]
    Sleigh(#[from] libsla::Error),

    #[error(transparent)]
    SymbolicValue(#[from] TryFromPcodeValueError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("syscall {syscall_num} is not handled")]
    UnhandledSyscall { syscall_num: u32 },

    #[error("a call with noreturn was completed")]
    Exit(i32),
}

pub type Result<T> = std::result::Result<T, Error>;

// https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl
#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum Syscall {
    Read = 0,
    Write = 1,
    Poll = 7,
    Mmap = 9,
    Mprotect = 10,
    Munmap = 11,
    Brk = 12,
    RtSigAction = 13,
    RtSigProcMask = 14,
    SigAltStack = 131,
    ArchPrctl = 158,
    SetTidAddress = 218,
    ExitGroup = 231,
    Ppoll = 271,
}

#[repr(i64)]
enum Errno {
    Einval = 22,
}

// https://github.com/torvalds/linux/blob/master/arch/x86/include/uapi/asm/prctl.h
#[repr(u32)]
pub enum ArchPrctlOp {
    SetFS = 0x1002,
}

// See https://man7.org/linux/man-pages/man2/syscall.2.html#NOTES
// for approriate details for various architectures
#[derive(Debug, Clone)]
pub struct LinuxArchConfig {
    pub syscall_num_register: String,
    pub arg_registers: [String; 6],
    pub return_register: String,
    pub syscall_map: std::collections::BTreeMap<u32, Syscall>,
}

#[derive(Debug, Clone)]
pub struct LinuxKernel {
    mmap_pages: BTreeSet<u64>,

    // https://github.com/torvalds/linux/blob/master/mm/nommu.c#L379
    brk_range: Range<u64>,
    brk: u64,

    exit_status: Option<i32>,
    arch_config: LinuxArchConfig,
}

impl Default for LinuxKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(Debug)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

// https://github.com/lsds/musl/blob/master/arch/sh/ksigaction.h
#[repr(C)]
struct SigAction {
    handler: u64,
    flags: u64,
    restorer: u64,
    mask: [u32; 2],
}

#[repr(C)]
struct Stack {
    /// Base address of stack
    ss_sp: u64,

    /// Flags
    ss_flags: i32,

    /// Number of bytes in stack
    ss_size: u64,
}

#[repr(i32)]
enum StackFlags {
    Disable = 0x2,
}

// https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/mman-common.h
#[repr(u64)]
enum MmapFlags {
    Fixed = 0x10,
    Anonymous = 0x20,
}

impl Kernel for LinuxKernel {
    fn syscall(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut impl VarnodeDataStore,
    ) -> emulator::Result<ControlFlow> {
        self.syscall_internal(sleigh, memory)
            .map_err(|err| emulator::Error::DependencyError(Box::new(err)))
    }
}

impl LinuxKernel {
    pub fn new() -> Self {
        Self::with_config(LinuxArchConfig {
            syscall_num_register: "EAX".to_owned(),
            arg_registers: [
                "RDI".to_owned(),
                "RSI".to_owned(),
                "RDX".to_owned(),
                "R10".to_owned(),
                "R8".to_owned(),
                "R9".to_owned(),
            ],
            return_register: "RAX".to_owned(),
            syscall_map: Default::default(),
        })
    }

    pub fn with_config(arch_config: LinuxArchConfig) -> Self {
        let brk_range = 0x9000000000..0xA000000000;
        Self {
            brk: brk_range.start,
            brk_range,
            mmap_pages: Default::default(),
            exit_status: None,
            arch_config,
        }
    }

    pub fn exit_status(&self) -> Option<i32> {
        self.exit_status
    }

    pub fn syscall_internal<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let syscall_num = self.syscall_num(sleigh, memory)?;
        if !self.arch_config.syscall_map.is_empty() {
            let syscall = self
                .arch_config
                .syscall_map
                .get(&syscall_num)
                .copied()
                .ok_or(Error::UnhandledSyscall { syscall_num })?;

            return match syscall {
                Syscall::Write => self.write(sleigh, memory),
                Syscall::Mmap => self.mmap(sleigh, memory),
                Syscall::Mprotect => self.mprotect(sleigh, memory),
                Syscall::Munmap => self.munmap(sleigh, memory),
                Syscall::Poll => self.poll(sleigh, memory),
                Syscall::Brk => self.brk(sleigh, memory),
                Syscall::RtSigProcMask => self.rt_sigprocmask(sleigh, memory),
                Syscall::RtSigAction => self.rt_sigaction(sleigh, memory),
                Syscall::SigAltStack => self.sigaltstack(sleigh, memory),
                Syscall::ArchPrctl => self.arch_prctl(sleigh, memory),
                Syscall::SetTidAddress => self.set_tid_address(sleigh, memory),
                Syscall::ExitGroup => self.exit_group(sleigh, memory),
                Syscall::Ppoll => self.ppoll(sleigh, memory),
                _ => Err(Error::UnhandledSyscall { syscall_num }),
            };
        }

        match syscall_num {
            n if n == Syscall::Write as u32 => self.write(sleigh, memory),
            n if n == Syscall::Mmap as u32 => self.mmap(sleigh, memory),
            n if n == Syscall::Mprotect as u32 => self.mprotect(sleigh, memory),
            n if n == Syscall::Munmap as u32 => self.munmap(sleigh, memory),
            n if n == Syscall::Poll as u32 => self.poll(sleigh, memory),
            n if n == Syscall::Brk as u32 => self.brk(sleigh, memory),
            n if n == Syscall::RtSigProcMask as u32 => self.rt_sigprocmask(sleigh, memory),
            n if n == Syscall::RtSigAction as u32 => self.rt_sigaction(sleigh, memory),
            n if n == Syscall::SigAltStack as u32 => self.sigaltstack(sleigh, memory),
            n if n == Syscall::ArchPrctl as u32 => self.arch_prctl(sleigh, memory),
            n if n == Syscall::SetTidAddress as u32 => self.set_tid_address(sleigh, memory),
            n if n == Syscall::ExitGroup as u32 => self.exit_group(sleigh, memory),
            _ => Err(Error::UnhandledSyscall { syscall_num }),
        }
    }

    fn ppoll<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // TODO Completely unimplemented and entirely guessing that return 0 is fine
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, 0u64)?;
        trace!("ppoll(...)");

        Ok(ControlFlow::NextInstruction)
    }

    fn poll<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let pfds: u64 = self.syscall_arg(sleigh, memory, 0)?;
        let nfds: i32 = self.syscall_arg(sleigh, memory, 1)?;
        let timeout: i32 = self.syscall_arg(sleigh, memory, 2)?;
        let ram = sleigh
            .address_space_by_name("ram")
            .expect("failed to find ram");
        let mut poll_fds: Vec<PollFd> = vec![];
        for pfds_offset in 0..nfds {
            let fds = VarnodeData::new(
                Address::new(ram.clone(), pfds + (8 * pfds_offset) as u64),
                std::mem::size_of::<PollFd>(),
            );
            let bytes = memory.read_value(&fds)?.try_into()?;

            // SAFETY: All byte combinations are valid
            poll_fds.push(unsafe {
                std::mem::transmute::<[u8; std::mem::size_of::<PollFd>()], PollFd>(bytes)
            });
        }

        // Number of fds with events. Say 0 for now
        let return_value = 0u64;
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, return_value)?;
        trace!("poll({poll_fds:?}, {nfds}, {timeout}) = {return_value}");

        Ok(ControlFlow::NextInstruction)
    }

    fn rt_sigaction<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // Signals not supported
        let signal: u32 = self.syscall_arg(sleigh, memory, 0)?;
        let new_sigaction_ptr: u64 = self.syscall_arg(sleigh, memory, 1)?;
        let old_sigaction_ptr: u64 = self.syscall_arg(sleigh, memory, 2)?;
        let size: u64 = self.syscall_arg(sleigh, memory, 3)?;

        if new_sigaction_ptr != 0 {
            warn!("Ignoring sigaction registration: {new_sigaction_ptr:#x}");
        }

        if old_sigaction_ptr != 0 {
            let ram = sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram");
            let old_sigaction_varnode = VarnodeData::new(
                Address::new(ram.clone(), old_sigaction_ptr),
                std::mem::size_of::<SigAction>(),
            );

            // All zeros means default handler with no special flags set
            let old_sigaction = [0u8; std::mem::size_of::<SigAction>()];
            memory.write_value(&old_sigaction_varnode, old_sigaction)?;
        }

        let return_value = 0u64;
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, return_value)?;

        trace!(
            "rt_sigaction({signal}, {new_sigaction_ptr:#x}, {old_sigaction_ptr:#x}, {size}) = {return_value}"
        );
        Ok(ControlFlow::NextInstruction)
    }

    fn rt_sigprocmask<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let _how: u32 = self.syscall_arg(sleigh, memory, 0)?;
        let _set: u64 = self.syscall_arg(sleigh, memory, 1)?;
        let oldset: u64 = self.syscall_arg(sleigh, memory, 2)?;
        let sigsetsize: u64 = self.syscall_arg(sleigh, memory, 3)?;

        if oldset != 0 {
            // Previous value must be stored here
            //
            // The value on x86 is 64-bit mask
            // https://github.com/torvalds/linux/blob/master/arch/x86/include/uapi/asm/signal.h#L16
            assert_eq!(sigsetsize, 8, "sigsetsize should be 8 bytes");

            let ram = sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram");
            memory.write_value(&VarnodeData::new(Address::new(ram, oldset), 8), 0u64)?;
        }

        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, 0u64)?;
        trace!("rt_sigprocmask(...)");
        Ok(ControlFlow::NextInstruction)
    }

    fn set_tid_address<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // https://www.man7.org/linux/man-pages/man2/set_tid_address.2.html
        // Multithreading not supported. Ignore this and return TID = 0
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, 0u64)?;
        trace!("set_tid_address(...)");
        Ok(ControlFlow::NextInstruction)
    }

    fn sigaltstack<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let new_ss: u64 = self.syscall_arg(sleigh, memory, 0)?;
        let old_ss: u64 = self.syscall_arg(sleigh, memory, 1)?;

        if new_ss != 0 {
            warn!("WARNING: Ignoring new signal alt stack");
        }

        if old_ss != 0 {
            let ram = sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram");

            let stack = Stack {
                ss_sp: 0,
                ss_flags: StackFlags::Disable as i32,
                ss_size: 0,
            };

            let stack: [u8; std::mem::size_of::<Stack>()] = unsafe { std::mem::transmute(stack) };

            memory.write_value(
                &VarnodeData::new(Address::new(ram, old_ss), std::mem::size_of::<Stack>()),
                stack,
            )?;
        }

        trace!("sigaltstack(...)");
        Ok(ControlFlow::NextInstruction)
    }

    fn arch_prctl<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // https://github.com/torvalds/linux/blob/master/arch/x86/include/uapi/asm/prctl.h
        let op: u32 = self.syscall_arg(sleigh, memory, 0)?;
        let addr: u64 = self.syscall_arg(sleigh, memory, 1)?;
        match op {
            n if n == ArchPrctlOp::SetFS as u32 => {
                // Set FS base address
                // In Ghidra this is modeled by the (fake) register FS_OFFSET
                let fs = sleigh.register_from_name("FS_OFFSET")?;
                memory.write_value(&fs, addr)?;

                // Write 0 on success
                let return_value = 0u64;
                let return_register =
                    sleigh.register_from_name(&self.arch_config.return_register)?;
                memory.write_value(&return_register, return_value)?;
                trace!("arch_prctl(ARCH_SET_FS, {addr:#x}) = {return_value}");
                Ok(ControlFlow::NextInstruction)
            }
            _ => Err(Error::UnhandledSyscall {
                syscall_num: Syscall::ArchPrctl as u32,
            }),
        }
    }

    fn mmap<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let addr: u64 = self.syscall_arg(sleigh, memory, 0)?;
        let len: u64 = self.syscall_arg(sleigh, memory, 1)?;
        let prot: u64 = self.syscall_arg(sleigh, memory, 2)?;
        let flags: u64 = self.syscall_arg(sleigh, memory, 3)?;
        let fd: i64 = self.syscall_arg(sleigh, memory, 4)?;
        let offset: u64 = self.syscall_arg(sleigh, memory, 5)?;

        if flags & MmapFlags::Anonymous as u64 == 0 {
            error!("File mmap not emulated");
            return Err(Error::UnhandledSyscall {
                syscall_num: Syscall::Mmap as u32,
            });
        }

        let return_value = if len == 0 {
            // EINVAL (since Linux 2.6.12) if length was 0.
            -(Errno::Einval as i64)
        } else {
            let mmap_addr = if flags & MmapFlags::Fixed as u64 != 0 {
                // TODO What if addr is 0 here?
                addr
            } else {
                self.mmap_pages
                    .last()
                    .copied()
                    .map(|page| page + 4096)
                    .unwrap_or(0xA000000000)
            };

            // MAP_ANONYMOUS must zero initialize contents
            let ram = sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram");
            for i in 0..len {
                let offset = mmap_addr + i;
                let zero_addr = VarnodeData::new(Address::new(ram.clone(), offset), 1);
                memory.write_value(&zero_addr, 0x0u8)?;
                if offset.is_multiple_of(4096) {
                    self.mmap_pages.insert(offset);
                }
            }

            mmap_addr as i64
        };

        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, return_value)?;
        trace!(
            "mmap({addr:#x}, {len}, {prot:#016x}, {flags:#016x}, {fd}, {offset}) = {return_value:#x}"
        );
        Ok(ControlFlow::NextInstruction)
    }

    fn mprotect<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // Noop, protection not enforced
        let return_value = 0u64;
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, return_value)?;
        trace!("mprotect(...) = {return_value}");
        Ok(ControlFlow::NextInstruction)
    }

    fn munmap<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let addr: u64 = self.syscall_arg(sleigh, memory, 0)?;
        let len: u64 = self.syscall_arg(sleigh, memory, 1)?;

        match self.mmap_pages.get(&addr).copied() {
            Some(start) => {
                for rm_addr in (start..start + len).step_by(4096) {
                    self.mmap_pages.remove(&rm_addr);
                }
            }
            None => {
                // It is not an error if the indicated range does not contain any mapped pages
                // per https://www.man7.org/linux/man-pages/man2/mmap.2.html
            }
        }

        let return_value = 0u64;
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, return_value)?;

        trace!("munmap({addr:#x}, {len}) = {return_value}");
        Ok(ControlFlow::NextInstruction)
    }

    fn brk<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // Noop, protection not enforced
        let brk: u64 = self.syscall_arg(sleigh, memory, 0)?;

        // Update brk if in valid range
        if self.brk_range.contains(&brk) {
            let ram = sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram");
            for addr in self.brk..=brk {
                // Allocating
                memory.write_value(&VarnodeData::new(Address::new(ram.clone(), addr), 1), 0x0u8)?;
            }

            self.brk = brk;
        }

        let return_value = self.brk;
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, return_value)?;

        trace!("brk({brk:08x}) = {return_value:08x}");
        Ok(ControlFlow::NextInstruction)
    }

    fn write<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        use std::io::Write;
        let fd: u64 = self.syscall_arg(sleigh, memory, 0)?;
        let buf: u64 = self.syscall_arg(sleigh, memory, 1)?;
        let count: u64 = self.syscall_arg(sleigh, memory, 2)?;

        let ram = sleigh
            .address_space_by_name("ram")
            .expect("failed to find ram");
        let target = VarnodeData::new(Address::new(ram, buf), count as usize);
        let bytes: Vec<u8> = memory.read_value(&target)?.try_into()?;

        match fd {
            1 => {
                let mut stdout = std::io::stdout();
                stdout.write_all(&bytes)?;
            }
            2 => {
                let mut stderr = std::io::stderr();
                stderr.write_all(&bytes)?;
            }
            _ => {
                //println!("WARNING: Writing to {fd} not supported");
                return Err(Error::UnhandledSyscall {
                    syscall_num: Syscall::Write as u32,
                });
            }
        }

        let return_value = count;
        let return_register = sleigh.register_from_name(&self.arch_config.return_register)?;
        memory.write_value(&return_register, return_value)?;

        trace!("write({fd}, {buf:#x}, {count}) = {return_value}");
        Ok(ControlFlow::NextInstruction)
    }

    fn exit_group<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &M,
    ) -> Result<ControlFlow> {
        let status: i32 = self.syscall_arg(sleigh, memory, 0)?;
        trace!("exitgroup({status})");
        self.exit_status = Some(status);
        Ok(ControlFlow::Halt)
    }

    fn syscall_num<M: VarnodeDataStore>(&self, sleigh: &impl Sleigh, memory: &M) -> Result<u32> {
        let syscall_reg = sleigh.register_from_name(&self.arch_config.syscall_num_register)?;
        let pcode_value = memory.read_value(&syscall_reg)?;
        Ok(u32::try_from(pcode_value)?)
    }

    fn syscall_arg<M, T>(&self, sleigh: &impl Sleigh, memory: &M, arg: usize) -> Result<T>
    where
        M: VarnodeDataStore,
        T: TryFrom<PcodeValue<M::Value>, Error = TryFromPcodeValueError>,
    {
        self.syscall_arg_from_register(sleigh, memory, &self.arch_config.arg_registers[arg])
    }

    fn syscall_arg_from_register<M, T>(
        &self,
        sleigh: &impl Sleigh,
        memory: &M,
        register_name: &str,
    ) -> Result<T>
    where
        M: VarnodeDataStore,
        T: TryFrom<PcodeValue<M::Value>, Error = TryFromPcodeValueError>,
    {
        let mut register = sleigh.register_from_name(register_name)?;

        // Truncate the read if we want less than the register holds
        if std::mem::size_of::<T>() < register.size {
            register.size = std::mem::size_of::<T>();
        }

        Ok(T::try_from(memory.read_value(&register)?)?)
    }
}
