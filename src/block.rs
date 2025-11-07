#![allow(dead_code)]
use burn::prelude::*;

use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Relu};

const CONV2_KERNEL_SIZE: [usize; 2] = [3, 3];
const MAX_POOL_KERNEL_SIZE: [usize; 2] = [2, 2];
const MAX_POOL_STRIDE_SIZE: [usize; 2] = [2, 2];

/// Generic Module block used within both the `EncoderBlock` & `DecoderBlock`.
#[derive(Debug, Module)]
pub struct Conv2dBlock<B: Backend> {
    conv_1: Conv2d<B>,
    conv_2: Conv2d<B>,
    norm_1: BatchNorm<B>,
    norm_2: BatchNorm<B>,
    activations: Relu,
    dropout: Option<Dropout>,
}

/// Conv2dBlock config to initialize `Conv2dBlock`
#[derive(Debug, Config)]
pub struct Conv2dBlockConfig {
    pub in_channel: usize,
    pub out_channel: usize,
    pub mid_channel: Option<usize>,
    pub dropout: Option<f64>,
}

impl From<(usize, usize)> for Conv2dBlockConfig {
    fn from(value: (usize, usize)) -> Self {
        let (in_channel, out_channel) = value;
        Self::new(in_channel, out_channel)
    }
}

impl Conv2dBlockConfig {
    /// Initializes a new Conv2dBlock module.
    pub fn init<B: Backend>(self, device: &B::Device) -> Conv2dBlock<B> {
        let Conv2dBlockConfig {
            in_channel,
            out_channel,
            mid_channel,
            dropout,
        } = self;

        let mid_channel = if let Some(channel) = mid_channel {
            channel
        } else {
            out_channel
        };

        let dropout = if let Some(drop) = dropout {
            Some(DropoutConfig::new(drop).init())
        } else {
            None
        };

        Conv2dBlock {
            conv_1: Conv2dConfig::new([in_channel, mid_channel], CONV2_KERNEL_SIZE)
                .with_padding(nn::PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            conv_2: Conv2dConfig::new([mid_channel, out_channel], CONV2_KERNEL_SIZE)
                .with_padding(nn::PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            norm_1: BatchNormConfig::new(mid_channel).init(device),
            norm_2: BatchNormConfig::new(out_channel).init(device),
            activations: Relu::new(),
            dropout,
        }
    }
}

impl<B: Backend> Conv2dBlock<B> {
    #[inline]
    /// Applies the forward pass on the input tensor.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x: Tensor<B, 4> = self.conv_1.forward(input.clone());
        x = self.norm_1.forward(x);
        x = self.activations.forward(x);
        x = self.conv_2.forward(x);
        x = self.norm_2.forward(x);
        x = self.activations.forward(x);
        // allow for drop out.
        if let Some(ref dropout) = self.dropout {
            dropout.forward(x)
        } else {
            x
        }
    }

    #[inline]
    /// Inference forward pass through Module.
    pub fn inference_forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x: Tensor<B, 4> = self.conv_1.forward(input);
        x = self.norm_1.forward(x);
        x = self.activations.forward(x);
        x = self.conv_2.forward(x);
        x = self.norm_2.forward(x);
        self.activations.forward(x)
    }
}

#[derive(Debug, Module)]
pub struct EncoderBlock<B: Backend> {
    core: Conv2dBlock<B>,
    pool: MaxPool2d,
}

#[derive(Debug, Config)]
pub struct EncoderBlockConfig {
    config: Conv2dBlockConfig,
}

impl EncoderBlockConfig {
    fn init<B: Backend>(self, device: &B::Device) -> EncoderBlock<B> {
        EncoderBlock {
            core: self.config.init(device),
            pool: MaxPool2dConfig::new(MAX_POOL_KERNEL_SIZE)
                .with_strides(MAX_POOL_STRIDE_SIZE)
                .init(),
        }
    }
}

impl<B: Backend> EncoderBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.core.forward(input);
        self.pool.forward(x)
    }

    pub fn forward_ref(&self, input: &Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.core.forward(input.clone());
        self.pool.forward(x)
    }
}

///
#[derive(Debug, Module)]
pub struct DecoderBlock<B: Backend> {
    core: Conv2dBlock<B>,
    conv_t: ConvTranspose2d<B>,
}

///
#[derive(Debug, Config)]
pub struct DecoderBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl<B: Backend> DecoderBlock<B> {
    /// Applies the forward pass through decoder block.
    pub fn forward(&self, input: Tensor<B, 4>, skip: Tensor<B, 4>) -> Tensor<B, 4> {
        let up = self.conv_t.forward(input);
        // concat the features along the channel dimension
        // B x C x H x W
        let x = Tensor::cat(vec![skip, up], 1);
        self.core.forward(x)
    }
}

#[cfg(test)]
mod test_block {
    use super::*;

    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn test_block_conv2d() {
        let device = NdArrayDevice::Cpu;
        let model: Conv2dBlock<NdArray> = Conv2dBlockConfig::new(1, 10).init(&device);
        let input: Tensor<NdArray, 4> = Tensor::from_data([[[[1], [1], [1]]]], &device);
        let output = model.forward(input);

        println!("{output}");
    }
}

//
// pub(crate) struct DecoderBlock<B: Backend> {}
