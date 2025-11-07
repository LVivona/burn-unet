#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::{Tensor, backend::Backend};

mod block;
pub use block::{DecoderBlock, DecoderBlockConfig};

use crate::block::{Conv2dBlock, EncoderBlock, EncoderBlockConfig};

///
#[derive(Config, Debug)]
pub struct UnetConfig {
    num_channels: usize,
    num_classes: usize,
    dropout: Option<f64>,
}

impl UnetConfig {
    ///
    pub fn init<B: Backend>(self, device: &B::Device) -> Unet<B> {
        todo!()
    }
}

///
#[derive(Module, Debug)]
pub struct Unet<B: Backend> {
    inc: Conv2dBlock<B>,
    encoder_block_1: EncoderBlock<B>,
    encoder_block_2: EncoderBlock<B>,
    encoder_block_3: EncoderBlock<B>,
    encoder_block_4: EncoderBlock<B>,
    decoder_block_1: DecoderBlock<B>,
    decoder_block_2: DecoderBlock<B>,
    decoder_block_3: DecoderBlock<B>,
    decoder_block_4: DecoderBlock<B>,
    output: Conv2d<B>,
}

impl<B: Backend> Unet<B> {
    /// Applies the foward pass through the U-net.
    ///
    pub fn foward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.inc.forward(input);
        let x2 = self.encoder_block_1.forward_ref(&x1);
        let x3 = self.encoder_block_2.forward_ref(&x2);
        let x4 = self.encoder_block_3.forward_ref(&x3);
        let x5 = self.encoder_block_4.forward_ref(&x4);

        let mut x = self.decoder_block_1.forward(x5, x4);
        x = self.decoder_block_2.forward(x, x3);
        x = self.decoder_block_3.forward(x, x2);
        x = self.decoder_block_4.forward(x, x1);
        let logits = self.output.forward(x);
        logits
    }
}
