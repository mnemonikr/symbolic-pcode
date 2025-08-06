use std::ops::Range;

use crate::emulator::{self, ControlFlow};
use crate::kernel::Kernel;
use crate::mem::VarnodeDataStore;
use libsla::{Address, Sleigh, VarnodeData};
use pcode_ops::convert::{PcodeValue, TryFromPcodeValueError};
use pcode_ops::PcodeOps;

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

#[derive(Debug, Clone)]
pub struct LinuxKernel {
    mmap_pages: Vec<u64>,

    // https://github.com/torvalds/linux/blob/master/mm/nommu.c#L379
    brk_range: Range<u64>,
    brk: u64,

    exit_status: Option<i32>,
}

impl Default for LinuxKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
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
    OnStack = 0x1,
    Disable = 0x2,
}

// https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/mman-common.h
#[repr(u64)]
enum MmapFlags {
    Anonymous = 0x20,
    Stack = 0x020000,
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
        let brk_range = 0x9000000000..0xA000000000;
        Self {
            brk: brk_range.start,
            brk_range,
            mmap_pages: Default::default(),
            exit_status: None,
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

    fn poll<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let nfds: i32 = self.syscall_arg(sleigh, memory, 0)?;
        let pfds: u64 = self.syscall_arg(sleigh, memory, 1)?;
        let timeout: i32 = self.syscall_arg(sleigh, memory, 2)?;
        let ram = sleigh
            .address_space_by_name("ram")
            .expect("failed to find ram");
        for pfds_offset in 0..nfds {
            let fds = VarnodeData::new(
                Address::new(ram.clone(), pfds + (8 * pfds_offset) as u64),
                std::mem::size_of::<PollFd>(),
            );
            let bytes: [u8; std::mem::size_of::<PollFd>()] =
                PcodeValue::from(memory.read(&fds)?).try_into()?;

            // SAFETY: All byte combinations are valid
            let poll_fd: PollFd = unsafe { std::mem::transmute(bytes) };
            //println!("Polling {fd} with timeout {timeout}", fd = poll_fd.fd);
        }

        // Number of fds with events. Say 0 for now
        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(0u64))?;

        Ok(ControlFlow::NextInstruction)
    }

    fn rt_sigaction<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // Signals not supported
        let _signal: u32 = self.syscall_arg(sleigh, memory, 0)?;
        let new_sigaction_ptr: u64 = self.syscall_arg(sleigh, memory, 1)?;
        let old_sigaction_ptr: u64 = self.syscall_arg(sleigh, memory, 2)?;
        let _size: u64 = self.syscall_arg(sleigh, memory, 3)?;
        //println!(
        //    "rt_sigaction({signal}, {new_sigaction_ptr:#08x}, {old_sigaction_ptr:#08x}, {size})"
        //);

        if new_sigaction_ptr != 0 {
            //println!("WARNING: Ignoring sigaction registration");
        }

        if old_sigaction_ptr != 0 {
            let ram = sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram");
            let old_sigaction_ptr = VarnodeData::new(
                Address::new(ram.clone(), old_sigaction_ptr),
                std::mem::size_of::<SigAction>(),
            );

            // All zeros means default handler with no special flags set
            let old_sigaction = [0u8; std::mem::size_of::<SigAction>()];
            memory.write(
                &old_sigaction_ptr,
                old_sigaction
                    .into_iter()
                    .map(<M::Value as PcodeOps>::Byte::from)
                    .collect(),
            )?;
        }

        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(0u64))?;
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
            memory.write(
                &VarnodeData::new(Address::new(ram, oldset), 8),
                M::Value::from_le(0u64),
            )?;
        }

        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(0u64))?;
        Ok(ControlFlow::NextInstruction)
    }

    fn set_tid_address<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // https://www.man7.org/linux/man-pages/man2/set_tid_address.2.html
        // Multithreading not supported. Ignore this and return TID = 0
        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(0u64))?;
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
            //println!("WARNING: Ignoring new signal alt stack");
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
            memory.write(
                &VarnodeData::new(Address::new(ram, old_ss), std::mem::size_of::<Stack>()),
                stack
                    .into_iter()
                    .map(<M::Value as PcodeOps>::Byte::from)
                    .collect(),
            )?;
        }

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
                memory.write(&fs, M::Value::from_le(addr))?;

                // Write 0 on success
                let rax = sleigh.register_from_name("RAX")?;
                memory.write(&rax, M::Value::from_le(0u64))?;
            }
            _ => {
                return Err(Error::UnhandledSyscall {
                    syscall_num: Syscall::ArchPrctl as u32,
                });
            }
        }

        Ok(ControlFlow::NextInstruction)
    }

    fn mmap<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let _addr: u64 = self.syscall_arg(sleigh, memory, 0)?;
        let len: u64 = self.syscall_arg(sleigh, memory, 1)?;
        let _prot: u64 = self.syscall_arg(sleigh, memory, 2)?;
        let flags: u64 = self.syscall_arg(sleigh, memory, 3)?;
        let _fd: u64 = self.syscall_arg(sleigh, memory, 4)?;
        let _offset: u64 = self.syscall_arg(sleigh, memory, 5)?;
        //println!("mmap({addr:016x}, {len}, {prot:016x}, {flags:016x}, {fd}, {offset})");

        if flags & MmapFlags::Anonymous as u64 == 0 {
            //println!("WARNING: File mmap not supported");
            return Err(Error::UnhandledSyscall {
                syscall_num: Syscall::Mmap as u32,
            });
        }

        // Kernel should use the addr as a hint but is not required to.
        let addr = self
            .mmap_pages
            .last()
            .copied()
            .map(|page| page.next_multiple_of(4096))
            .unwrap_or(0xA000000000);

        // Zero initialize contents
        let ram = sleigh
            .address_space_by_name("ram")
            .expect("failed to find ram");
        for i in 0..len {
            let offset = addr + i;
            let zero_addr = VarnodeData::new(Address::new(ram.clone(), offset), 1);
            memory.write(&zero_addr, M::Value::from_le(0u8))?;
            if offset % 4096 == 0 {
                self.mmap_pages.push(offset);
            }
        }

        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(addr))?;

        Ok(ControlFlow::NextInstruction)
    }

    fn mprotect<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        // Noop, protection not enforced
        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(0u64))?;
        Ok(ControlFlow::NextInstruction)
    }

    fn munmap<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut M,
    ) -> Result<ControlFlow> {
        let addr: u64 = self.syscall_arg(sleigh, memory, 0)?;
        let len: u64 = self.syscall_arg(sleigh, memory, 1)?;
        //println!("munmap({addr:016x}, {len})");

        match self.mmap_pages.binary_search(&addr) {
            Ok(start) => {
                // Unmap this page and possibly more
                let num_pages = (len / 4096) as usize;
                if start + num_pages > self.mmap_pages.len() {
                    // Length would unmap too many pages
                    let rax = sleigh.register_from_name("RAX")?;
                    memory.write(&rax, M::Value::from_le(-(Errno::Einval as i64)))?;
                    return Ok(ControlFlow::NextInstruction);
                }

                let mut rm_addr = addr;
                for i in 0..num_pages {
                    // Ensure that these pages match expected addr values
                    if rm_addr != self.mmap_pages[start + i] {
                        // Length is too large, would unmap unrelated page
                        let rax = sleigh.register_from_name("RAX")?;
                        memory.write(&rax, M::Value::from_le(-(Errno::Einval as i64)))?;
                        return Ok(ControlFlow::NextInstruction);
                    }
                    rm_addr = rm_addr.next_multiple_of(4096);
                }

                // Everything is good!
                // Using a Vec here makes this SLOW. Consider a BtreeSet instead since we are just
                // tracking whether a page is allocated or not.
                for _ in 0..num_pages {
                    self.mmap_pages.remove(start);
                }
            }
            Err(_) => {
                // Address is not mapped
                let rax = sleigh.register_from_name("RAX")?;
                memory.write(&rax, M::Value::from_le(-(Errno::Einval as i64)))?;
                return Ok(ControlFlow::NextInstruction);
            }
        }

        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(0u64))?;
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
                memory.write(
                    &VarnodeData::new(Address::new(ram.clone(), addr), 1),
                    M::Value::from_le(0u8),
                )?;
            }

            self.brk = brk;
        }

        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(self.brk))?;
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
        let bytes: Vec<u8> = PcodeValue::from(memory.read(&target)?).try_into()?;

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

        let rax = sleigh.register_from_name("RAX")?;
        memory.write(&rax, M::Value::from_le(count))?;
        Ok(ControlFlow::NextInstruction)
    }

    fn exit_group<M: VarnodeDataStore>(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &M,
    ) -> Result<ControlFlow> {
        let status: i32 = self.syscall_arg(sleigh, memory, 0)?;
        self.exit_status = Some(status);
        Ok(ControlFlow::Halt)
    }

    fn syscall_num<M: VarnodeDataStore>(&self, sleigh: &impl Sleigh, memory: &M) -> Result<u32> {
        let eax = sleigh.register_from_name("EAX")?;
        let pcode_value = PcodeValue::from(memory.read(&eax)?);
        Ok(u32::try_from(pcode_value)?)
    }

    fn syscall_arg<M, T>(&self, sleigh: &impl Sleigh, memory: &M, arg: usize) -> Result<T>
    where
        M: VarnodeDataStore,
        T: TryFrom<PcodeValue<M::Value>, Error = TryFromPcodeValueError>,
    {
        // For x86-64
        let registers = ["RDI", "RSI", "RDX", "R10", "R8", "R9"];
        self.syscall_arg_from_register(sleigh, memory, registers[arg])
    }

    fn syscall_arg_from_register<M, T>(
        &self,
        sleigh: &impl Sleigh,
        memory: &M,
        register_name: &'static str,
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

        Ok(T::try_from(PcodeValue::from(memory.read(&register)?))?)
    }
}
