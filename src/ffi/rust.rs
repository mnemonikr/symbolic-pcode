use super::api;
use super::sys;
use cxx::{CxxString, CxxVector};

pub struct RustAssemblyEmit<'a>(pub &'a mut dyn api::AssemblyEmit);

impl<'a> RustAssemblyEmit<'a> {
    pub fn dump(&mut self, address: &sys::Address, mnemonic: &CxxString, body: &CxxString) {
        self.0.dump(address, mnemonic, body);
    }
}

pub struct RustLoadImage<'a>(pub &'a dyn api::LoadImage);

impl<'a> RustLoadImage<'a> {
    /// * `ptr` must have `len` elements and be safe to dereference if `len > 0`
    pub unsafe fn load_fill(
        &self,
        ptr: *mut u8,
        len: i32,
        addr: &sys::Address,
    ) -> Result<(), String> {
        let mut buffer = std::slice::from_raw_parts_mut(ptr, len as usize);
        self.0.load_fill(&mut buffer, &addr)
    }
}

pub struct RustPcodeEmit<'a>(pub &'a mut dyn api::PcodeEmit);

impl<'a> RustPcodeEmit<'a> {
    /// # SAFETY:
    /// * `output` must be safe to dereference if non-null. Note that this value MAY be null.
    pub unsafe fn dump(
        &mut self,
        addr: &sys::Address,
        op_code: sys::OpCode,
        output: *mut sys::VarnodeData, // TODO make const?
        inputs: &CxxVector<sys::VarnodeData>,
    ) {
        let output = if !output.is_null() {
            Some(&*output)
        } else {
            None
        };

        self.0.dump(&addr, op_code, output, inputs);
    }
}
