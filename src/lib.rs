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
            inc: Conv2dBlockConfig::new(num_channels, DEFAULT_BLOCK_1_FEATRUES).init(device),
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
            decoder_block_2: DecoderBlockConfig::from((
                DEFAULT_BLOCK_4_FEATRUES,
                DEFAULT_BLOCK_3_FEATRUES,
            ))
            .init(device),
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
    /// Applies the forward pass through the U-net.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // NOTE: Since we need the enocder block output during the decoding
        // we have to preform the clone through ref, if done right in the backend
        // our memory shouldn't increase too much as we would be only gettiing
        // a unmutable reference to the address the tensor is pointing to
        // but it not this doubles our memeory until we can drop the tensors
        // within the skip connection.
        let x1 = self.inc.forward(input);
        let (x, x2) = self.encoder_block_1.forward(x1.clone());
        let (x, x3) = self.encoder_block_2.forward(x);
        let (x, x4) = self.encoder_block_3.forward(x);
        let (x, x5) = self.encoder_block_4.forward(x);

        let mut x = self.decoder_block_1.forward(x, x5);
        x = self.decoder_block_2.forward(x, x4);
        x = self.decoder_block_3.forward(x, x3);
        x = self.decoder_block_4.forward(x, x2);
        self.output.forward(x)
    }
}

#[cfg(test)]
mod unet_test {

    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Shape, Tensor};

    use crate::{Unet, UnetConfig};

    #[test]
    fn test_unet_single_batch() {
        let device = NdArrayDevice::Cpu;
        let model: Unet<NdArray> = UnetConfig::new(1, 1).init(&device);

        let input = Tensor::ones(Shape::from([1, 1, 221, 222]), &device);

        let output = model.forward(input);
        println!("{:?}", output.shape())
    }
}
