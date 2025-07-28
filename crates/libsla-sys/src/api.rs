use super::sys;
use cxx::{CxxString, CxxVector};

pub trait PcodeEmit {
    fn dump(
        &mut self,
        address: &sys::Address,
        op_code: sys::OpCode,
        output_variable: Option<&sys::VarnodeData>,
        input_variables: &CxxVector<sys::VarnodeData>,
    );
}

pub trait AssemblyEmit {
    fn dump(&mut self, address: &sys::Address, mnemonic: &CxxString, operands: &CxxString);
}

pub trait LoadImage {
    fn load_fill(&self, data: &mut [u8], address: &sys::Address) -> Result<(), String>;
}
