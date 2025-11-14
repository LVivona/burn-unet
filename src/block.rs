#![allow(dead_code)]
use burn::prelude::*;

use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Relu};

const CONV2_KERNEL_SIZE: [usize; 2] = [3, 3];
const MAX_POOL_KERNEL_SIZE: [usize; 2] = [2, 2];
const MAX_POOL_STRIDE_SIZE: [usize; 2] = [2, 2];

/// Applies Double Conv2d Module block used within both the `EncoderBlock` & `DecoderBlock`.
///
/// Also refered as the [DoubleConv](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py).
#[derive(Debug, Module)]
pub struct Conv2dBlock<B: Backend> {
    /// First [2d convolution](burn::nn::Conv2d) layer.
    pub conv_1: Conv2d<B>,
    /// Second [2d convolution](burn::nn::Conv2d) layer.
    pub conv_2: Conv2d<B>,
    /// First Batch Normalization over the batch input.
    pub norm_1: BatchNorm<B>,
    /// Second Batch Normalization over the batch input.
    pub norm_2: BatchNorm<B>,
    /// Activation layer over `conv_i -> norm_i` where {i : 0 <= i <= 1}
    pub activations: Relu,
    /// Training Drouput layer saetting some random input element to 0.
    pub dropout: Option<Dropout>,
}

/// Configuation to create a Conv2dBlock, using [init function](Conv2dBlockConfig::init).
#[derive(Debug, Config)]
pub struct Conv2dBlockConfig {
    /// The number of input channels. `[batch, channels, ..]`
    pub in_channels: usize,
    /// The number of output channels `[batch, channels, ..]`
    pub out_channels: usize,
    /// The number of intermediate channels within hidden layer.
    pub mid_channels: Option<usize>,
    /// Does training requires dropout.
    pub dropout: Option<f64>,
}

impl From<(usize, usize)> for Conv2dBlockConfig {
    fn from(value: (usize, usize)) -> Self {
        let (in_channels, out_channels) = value;
        Self::new(in_channels, out_channels)
    }
}

impl Conv2dBlockConfig {
    /// Initializes a new Conv2dBlock module.
    pub fn init<B: Backend>(self, device: &B::Device) -> Conv2dBlock<B> {
        let Conv2dBlockConfig {
            in_channels,
            out_channels,
            mid_channels,
            dropout,
        } = self;

        let mid_channels = if let Some(channel) = mid_channels {
            channel
        } else {
            out_channels
        };

        let dropout = dropout.map(|d| DropoutConfig::new(d).init());

        Conv2dBlock {
            conv_1: Conv2dConfig::new([in_channels, mid_channels], CONV2_KERNEL_SIZE)
                .with_padding(nn::PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            conv_2: Conv2dConfig::new([mid_channels, out_channels], CONV2_KERNEL_SIZE)
                .with_padding(nn::PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            norm_1: BatchNormConfig::new(mid_channels).init(device),
            norm_2: BatchNormConfig::new(out_channels).init(device),
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

/// Applies a Encoder operation over the input tensors.
///
/// Should be created with [EncoderBlockConfig]
#[derive(Debug, Module)]
pub struct EncoderBlock<B: Backend> {
    /// Double Convolution over input.
    pub core: Conv2dBlock<B>,
    /// Pooling over output image.
    pub pool: MaxPool2d,
}

#[derive(Debug, Config)]
pub struct EncoderBlockConfig {
    /// Inner Double [Conv2dBlock] config.
    pub config: Conv2dBlockConfig,
}

impl From<(usize, usize)> for EncoderBlockConfig {
    fn from(value: (usize, usize)) -> Self {
        let (in_channels, out_channels) = value;
        EncoderBlockConfig::new(Conv2dBlockConfig::new(in_channels, out_channels))
    }
}

impl EncoderBlockConfig {
    /// Initialize a new [encoder](EncoderBlock) module.
    pub fn init<B: Backend>(self, device: &B::Device) -> EncoderBlock<B> {
        EncoderBlock {
            core: self.config.init(device),
            pool: MaxPool2dConfig::new(MAX_POOL_KERNEL_SIZE)
                .with_strides(MAX_POOL_STRIDE_SIZE)
                .init(),
        }
    }
}

impl<B: Backend> EncoderBlock<B> {
    /// Applies the forward pass through the encoder.
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.core.forward(input);
        (self.pool.forward(x.clone()), x)
    }
}

/// Applies a decoder operation over the input tensors.
///
/// Should be created with [DecoderBlockConfig]
#[derive(Debug, Module)]
pub struct DecoderBlock<B: Backend> {
    /// Output projection of concat connection and upsample.
    pub core: Conv2dBlock<B>,
    /// Upsample the convolution input.
    pub conv_t: ConvTranspose2d<B>,
}

/// Configuration to create a Decoder block, uisng [init function](DecoderBlockConfig::init)
#[derive(Debug, Config)]
pub struct DecoderBlockConfig {
    /// The number of input channels.
    pub in_channels: usize,
    /// The number of output channels.
    pub out_channels: usize,
}

impl DecoderBlockConfig {
    /// Initialize a new [decoder](DecoderBlock) module.
    pub fn init<B: Backend>(self, device: &B::Device) -> DecoderBlock<B> {
        let DecoderBlockConfig {
            in_channels,
            out_channels,
        } = self;
        DecoderBlock {
            core: Conv2dBlockConfig::new(in_channels, out_channels).init(device),
            conv_t: ConvTranspose2dConfig::new([in_channels, in_channels / 2], [2, 2])
                .with_stride([2, 2])
                .init(device),
        }
    }
}

impl From<(usize, usize)> for DecoderBlockConfig {
    fn from(value: (usize, usize)) -> Self {
        DecoderBlockConfig {
            in_channels: value.0,
            out_channels: value.1,
        }
    }
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
mod block_test {

    use super::*;

    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    use burn::Tensor;

    #[test]
    fn test_ndarray_backend() {
        let device = NdArrayDevice::Cpu;
        let model: Conv2dBlock<NdArray> = Conv2dBlockConfig::new(1, 64).init(&device);

        let input: Tensor<NdArray, 4> = Tensor::ones(Shape::from([1, 1, 224, 224]), &device);

        let logits = model.forward(input);
        let shape = logits.shape();
        let expected_shape = Shape::from([1, 5, 2, 2]);
        assert_eq!(shape, expected_shape);
    }
}
