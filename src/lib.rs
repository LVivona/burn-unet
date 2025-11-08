#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::{Tensor, backend::Backend};

mod block;
pub use block::{DecoderBlock, DecoderBlockConfig};

use crate::block::{Conv2dBlock, Conv2dBlockConfig, EncoderBlock, EncoderBlockConfig};

const DEFAULT_BLOCK_1_FEATRUES: usize = 64;
const DEFAULT_BLOCK_2_FEATRUES: usize = 128;
const DEFAULT_BLOCK_3_FEATRUES: usize = 258;
const DEFAULT_BLOCK_4_FEATRUES: usize = 512;
const DEFAULT_BLOCK_5_FEATRUES: usize = 1024;

/// Configuration to create a Unet block, uisng [init function](UnetConfig::init)
#[derive(Config, Debug)]
pub struct UnetConfig {
    /// The number of input channels
    pub num_channels: usize,
    /// The number of output classes.
    pub num_classes: usize,
}

impl UnetConfig {
    /// Initialize a new [unet](Unet) module.
    ///
    /// Inspired [Unet](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py) block.
    pub fn init<B: Backend>(self, device: &B::Device) -> Unet<B> {
        let UnetConfig {
            num_channels,
            num_classes,
            ..
        } = self;

        Unet {
            inc: Conv2dBlockConfig::new(num_channels, 64).init(device),
            encoder_block_1: EncoderBlockConfig::from((
                DEFAULT_BLOCK_1_FEATRUES,
                DEFAULT_BLOCK_2_FEATRUES,
            ))
            .init(device),
            encoder_block_2: EncoderBlockConfig::from((
                DEFAULT_BLOCK_2_FEATRUES,
                DEFAULT_BLOCK_3_FEATRUES,
            ))
            .init(device),
            encoder_block_3: EncoderBlockConfig::from((
                DEFAULT_BLOCK_3_FEATRUES,
                DEFAULT_BLOCK_4_FEATRUES,
            ))
            .init(device),
            encoder_block_4: EncoderBlockConfig::from((
                DEFAULT_BLOCK_4_FEATRUES,
                DEFAULT_BLOCK_5_FEATRUES,
            ))
            .init(device),
            decoder_block_1: DecoderBlockConfig::from((
                DEFAULT_BLOCK_5_FEATRUES,
                DEFAULT_BLOCK_4_FEATRUES,
            ))
            .init(device),
            decoder_block_2: DecoderBlockConfig::from((512, DEFAULT_BLOCK_3_FEATRUES)).init(device),
            decoder_block_3: DecoderBlockConfig::from((
                DEFAULT_BLOCK_3_FEATRUES,
                DEFAULT_BLOCK_2_FEATRUES,
            ))
            .init(device),
            decoder_block_4: DecoderBlockConfig::from((
                DEFAULT_BLOCK_2_FEATRUES,
                DEFAULT_BLOCK_1_FEATRUES,
            ))
            .init(device),
            output: Conv2dConfig::new([DEFAULT_BLOCK_1_FEATRUES, num_classes], [1, 1]).init(device),
        }
    }
}

/// Applies a Unet operation over the input tensors.
///
/// Should be created with [UnetConfig].
#[derive(Module, Debug)]
pub struct Unet<B: Backend> {
    /// Input projection block.
    pub inc: Conv2dBlock<B>,
    /// Intermideate encoder block.
    pub encoder_block_1: EncoderBlock<B>,
    /// Intermideate encoder block.
    pub encoder_block_2: EncoderBlock<B>,
    /// Intermideate encoder block.
    pub encoder_block_3: EncoderBlock<B>,
    /// Intermideate bottleneck block.
    pub encoder_block_4: EncoderBlock<B>,
    /// Intermideate decoder upsample block.
    pub decoder_block_1: DecoderBlock<B>,
    /// Intermideate decoder upsample block.
    pub decoder_block_2: DecoderBlock<B>,
    /// Intermideate decoder upsample block.
    pub decoder_block_3: DecoderBlock<B>,
    /// Final decoder block.
    pub decoder_block_4: DecoderBlock<B>,
    /// Ouput logits projection, shape `[channel, height, width]`.
    pub output: Conv2d<B>,
}

impl<B: Backend> Unet<B> {
    /// Applies the foward pass through the U-net.
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
